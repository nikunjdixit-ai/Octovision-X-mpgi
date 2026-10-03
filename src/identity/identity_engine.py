from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.types import FramePacket, TrackedPerson
from src.identity.enrollment import (
    EXPECTED_EMBEDDING_DIM,
    EmbeddingExtractorProtocol,
    StudentRegistry,
    normalize_l2,
    sanitize_roll_number,
)
from src.reid.gallery import parse_stream_track_uid

logger = logging.getLogger("trace.identity.engine")

STATUS_CONFIRMED = "CONFIRMED"
STATUS_TENTATIVE = "TENTATIVE"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_UNKNOWN = "UNKNOWN"


@dataclass
class IdentityDecision:
    """
    Represents the IdentityEngine decision for a single tracked person at a point in time.
    """
    stream_track_uid: str
    camera_id: str
    session_id: str
    track_id: int
    status: str  # "CONFIRMED", "TENTATIVE", "LOW_CONFIDENCE", "UNKNOWN"
    student_id: Optional[str] = None
    name: str = STATUS_UNKNOWN
    roll_number: Optional[str] = None
    similarity: float = 0.0
    candidate_margin: float = 0.0
    candidate_student_id: Optional[str] = None
    candidate_name: Optional[str] = None
    observations: int = 0
    confirmations: int = 0
    frame_id: int = 0
    inference_performed: bool = True
    conflict_demoted: bool = False
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.roll_number is not None:
            self.roll_number = sanitize_roll_number(self.roll_number)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("roll_number") is not None:
            d["roll_number"] = str(self.roll_number)
        return d


def assess_crop_quality(crop: Optional[np.ndarray]) -> Tuple[bool, float, str]:
    """
    Evaluates person crop quality before embedding extraction:
    - Minimum crop dimensions (height >= 24, width >= 16)
    - Aspect ratio sanity check
    - Blur / sharpness estimation via Laplacian variance
    Returns (is_valid, quality_score, reason).
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim != 3:
        return False, 0.0, "empty_crop"
    h, w = crop.shape[:2]
    if h < 24 or w < 16:
        return False, 0.0, "too_small"
    aspect = h / float(max(1, w))
    if aspect < 0.35 or aspect > 6.5:
        return False, 0.1, "extreme_aspect_ratio"
    try:
        import cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        lap_var = 50.0
    size_score = float(np.clip((h * w) / (160.0 * 80.0), 0.2, 1.0))
    sharp_score = float(np.clip(lap_var / 80.0, 0.25, 1.0))
    return True, round(0.5 * size_score + 0.5 * sharp_score, 4), "ok"


@dataclass
class _TrackIdentityState:
    """Internal per-track temporal state keyed by stream_track_uid."""
    stream_track_uid: str
    camera_id: str
    session_id: str
    track_id: int
    frames_seen: int = 0
    observations: int = 0
    last_inference_step: int = -999999
    vote_counts: Dict[str, int] = field(default_factory=dict)
    similarity_history: Dict[str, List[float]] = field(default_factory=dict)
    biometric_history: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = field(default_factory=list)
    confirmed_student_id: Optional[str] = None
    locked_student_id: Optional[str] = None
    last_seen_frame_id: int = 0
    last_decision: Optional[IdentityDecision] = None


class IdentityEngine:
    """
    Stateful person identity engine mapping tracked person crops/embeddings to enrolled students.

    Key features:
    - Never forces Top-1 match when similarity is below confirm_threshold.
    - Distinguishes CONFIRMED, TENTATIVE, LOW_CONFIDENCE, and UNKNOWN states.
    - Temporal stabilization requiring `min_confirmations` matching observations before confirmation.
    - Periodic frame-interval inference skipping to protect RTX 3050 4 GB VRAM.
    - Simultaneous active track conflict resolution within the same (camera_id, session_id).
    """

    def __init__(
        self,
        registry: StudentRegistry,
        reid_model: Optional[EmbeddingExtractorProtocol] = None,
        confirm_threshold: float = 0.70,
        low_confidence_threshold: float = 0.50,
        min_confirmations: int = 3,
        inference_interval_frames: int = 5,
        reverify_interval_frames: int = 30,
        active_track_ttl_frames: int = 30,
        min_margin: float = 0.015,
    ):
        if confirm_threshold <= 0.0 or confirm_threshold > 1.0:
            raise ValueError("confirm_threshold must be in (0.0, 1.0].")
        if low_confidence_threshold < 0.0 or low_confidence_threshold > confirm_threshold:
            raise ValueError("low_confidence_threshold must be in [0.0, confirm_threshold].")
        if min_confirmations < 1:
            raise ValueError("min_confirmations must be >= 1.")
        if inference_interval_frames < 1:
            raise ValueError("inference_interval_frames must be >= 1.")
        if reverify_interval_frames < 1:
            raise ValueError("reverify_interval_frames must be >= 1.")
        if min_margin < 0.0:
            raise ValueError("min_margin must be >= 0.0.")

        self.registry = registry
        self._reid_model = reid_model
        self.confirm_threshold = float(confirm_threshold)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.min_confirmations = int(min_confirmations)
        self.inference_interval_frames = int(inference_interval_frames)
        self.reverify_interval_frames = int(reverify_interval_frames)
        self.active_track_ttl_frames = int(active_track_ttl_frames)
        self.min_margin = float(min_margin)

        # Keyed by stream_track_uid
        self._track_states: Dict[str, _TrackIdentityState] = {}

        # Telemetry counters
        self.total_inferences: int = 0
        self.total_skipped_frames: int = 0
        self.embedding_latencies_ms: List[float] = []
        self.matching_latencies_ms: List[float] = []

    @property
    def reid_model(self) -> EmbeddingExtractorProtocol:
        if self._reid_model is not None:
            return self._reid_model
        return self.registry.reid_model

    def reset(self) -> None:
        """Clears all tracked identity states and telemetry counters across sessions."""
        self._track_states.clear()
        self.total_inferences = 0
        self.total_skipped_frames = 0
        self.embedding_latencies_ms.clear()
        self.matching_latencies_ms.clear()

    def get_track_state(self, stream_track_uid: str) -> Optional[IdentityDecision]:
        state = self._track_states.get(stream_track_uid)
        return state.last_decision if state else None

    def _get_or_create_state(
        self,
        stream_track_uid: str,
        camera_id: Optional[str] = None,
        session_id: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> _TrackIdentityState:
        if stream_track_uid not in self._track_states:
            parsed = parse_stream_track_uid(stream_track_uid)
            cam = camera_id or parsed.get("camera_id") or "CAM_01"
            sess = session_id or parsed.get("session_id") or "SESSION_01"
            tid = track_id if track_id is not None else (parsed.get("track_id") or 0)
            self._track_states[stream_track_uid] = _TrackIdentityState(
                stream_track_uid=stream_track_uid,
                camera_id=str(cam),
                session_id=str(sess),
                track_id=int(tid),
            )
        return self._track_states[stream_track_uid]

    def _should_run_inference(self, state: _TrackIdentityState) -> bool:
        """
        Determines whether OSNet feature extraction should run on the current frame for this track.
        - Always runs on the very first frame observation of a track (frames_seen == 1).
        - If unconfirmed: runs every `inference_interval_frames`.
        - If already CONFIRMED: runs every `reverify_interval_frames`.
        """
        if state.observations == 0:
            return True

        steps_since_last = state.frames_seen - state.last_inference_step
        if state.confirmed_student_id is not None:
            return steps_since_last >= self.reverify_interval_frames
        else:
            return steps_since_last >= self.inference_interval_frames

    def evaluate_embedding(
        self,
        stream_track_uid: str,
        embedding: np.ndarray,
        frame_id: int = 0,
        camera_id: Optional[str] = None,
        session_id: Optional[str] = None,
        track_id: Optional[int] = None,
        resolve_conflicts: bool = True,
    ) -> IdentityDecision:
        """
        Directly evaluates a 512-D OSNet embedding for a given `stream_track_uid`,
        updating temporal stabilization state and returning an IdentityDecision.
        """
        t0 = time.perf_counter()
        state = self._get_or_create_state(stream_track_uid, camera_id, session_id, track_id)
        state.frames_seen += 1
        state.observations += 1
        state.last_inference_step = state.frames_seen
        state.last_seen_frame_id = frame_id

        norm_emb = normalize_l2(embedding, expected_dim=EXPECTED_EMBEDDING_DIM)
        from src.reid.reid_model import unpack_hybrid_embedding

        is_bio, sf, hd, bd, q = unpack_hybrid_embedding(norm_emb)
        has_bio_gallery = getattr(self.registry, "_has_biometric_gallery", False)

        # Evaluate the current frame's normalized embedding without averaging & re-normalizing
        # noisy track SFace vectors (which previously inflated cosine similarity by 1 / ||mean(sf)||).
        if is_bio and q >= 0.40:
            state.biometric_history.append((sf, hd, bd, q))

        candidates = self.registry.query_identity(norm_emb)
        match_ms = (time.perf_counter() - t0) * 1000.0
        self.matching_latencies_ms.append(match_ms)

        if not candidates:
            decision = IdentityDecision(
                stream_track_uid=state.stream_track_uid,
                camera_id=state.camera_id,
                session_id=state.session_id,
                track_id=state.track_id,
                status=STATUS_UNKNOWN,
                student_id=None,
                name=STATUS_UNKNOWN,
                roll_number=None,
                similarity=0.0,
                observations=state.observations,
                confirmations=0,
                frame_id=frame_id,
                inference_performed=True,
                latency_ms=round(match_ms, 3),
            )
            state.last_decision = decision
            return decision

        top = candidates[0]
        top_sim = float(top["similarity"])
        top_sid = str(top["student_id"])
        top_name = str(top["name"])

        candidate_margin = (
            float(top_sim - candidates[1]["similarity"])
            if len(candidates) > 1
            else 1.0
        )

        if has_bio_gallery:
            eff_margin = max(self.min_margin, 0.04)
            if is_bio and q >= 0.40 and top_sim >= self.confirm_threshold and candidate_margin >= eff_margin:
                state.vote_counts[top_sid] = state.vote_counts.get(top_sid, 0) + 1
                state.similarity_history.setdefault(top_sid, []).append(top_sim)

            if state.vote_counts:
                best_sid = max(
                    state.vote_counts.keys(),
                    key=lambda s: (state.vote_counts[s], np.mean(state.similarity_history[s])),
                )
                best_votes = state.vote_counts[best_sid]
                mean_sim = float(np.mean(state.similarity_history[best_sid]))
                best_profile = self.registry.get_student(best_sid)
                total_bio = max(1, len(state.biometric_history))
                vote_ratio = best_votes / float(total_bio)

                consistency_ok = (
                    (best_votes >= max(self.min_confirmations, 5) and vote_ratio >= 0.25 and mean_sim >= 0.735)
                    or (best_votes >= self.min_confirmations and vote_ratio >= 0.50 and mean_sim >= 0.760)
                )

                if consistency_ok and best_profile is not None:
                    state.confirmed_student_id = best_sid
                    state.locked_student_id = best_sid
                    decision = IdentityDecision(
                        stream_track_uid=state.stream_track_uid,
                        camera_id=state.camera_id,
                        session_id=state.session_id,
                        track_id=state.track_id,
                        status=STATUS_CONFIRMED,
                        student_id=best_sid,
                        name=best_profile.name,
                        roll_number=best_profile.roll_number,
                        similarity=round(mean_sim, 4),
                        candidate_margin=round(candidate_margin, 4),
                        candidate_student_id=best_sid,
                        candidate_name=best_profile.name,
                        observations=state.observations,
                        confirmations=best_votes,
                        frame_id=frame_id,
                        inference_performed=True,
                        latency_ms=round(match_ms, 3),
                    )
                else:
                    state.confirmed_student_id = None
                    state.locked_student_id = None
                    decision = IdentityDecision(
                        stream_track_uid=state.stream_track_uid,
                        camera_id=state.camera_id,
                        session_id=state.session_id,
                        track_id=state.track_id,
                        status=STATUS_TENTATIVE,
                        student_id=None,
                        name=STATUS_TENTATIVE,
                        roll_number=None,
                        similarity=round(mean_sim, 4),
                        candidate_margin=round(candidate_margin, 4),
                        candidate_student_id=best_sid,
                        candidate_name=best_profile.name if best_profile else top_name,
                        observations=state.observations,
                        confirmations=best_votes,
                        frame_id=frame_id,
                        inference_performed=True,
                        latency_ms=round(match_ms, 3),
                    )
            elif top_sim >= self.low_confidence_threshold:
                decision = IdentityDecision(
                    stream_track_uid=state.stream_track_uid,
                    camera_id=state.camera_id,
                    session_id=state.session_id,
                    track_id=state.track_id,
                    status=STATUS_LOW_CONFIDENCE,
                    student_id=None,
                    name=STATUS_LOW_CONFIDENCE,
                    roll_number=None,
                    similarity=round(top_sim, 4),
                    candidate_margin=round(candidate_margin, 4),
                    candidate_student_id=top_sid,
                    candidate_name=top_name,
                    observations=state.observations,
                    confirmations=0,
                    frame_id=frame_id,
                    inference_performed=True,
                    latency_ms=round(match_ms, 3),
                )
            else:
                decision = IdentityDecision(
                    stream_track_uid=state.stream_track_uid,
                    camera_id=state.camera_id,
                    session_id=state.session_id,
                    track_id=state.track_id,
                    status=STATUS_UNKNOWN,
                    student_id=None,
                    name=STATUS_UNKNOWN,
                    roll_number=None,
                    similarity=round(top_sim, 4),
                    candidate_margin=round(candidate_margin, 4),
                    candidate_student_id=None,
                    candidate_name=None,
                    observations=state.observations,
                    confirmations=0,
                    frame_id=frame_id,
                    inference_performed=True,
                    latency_ms=round(match_ms, 3),
                )
        elif state.locked_student_id is not None:
            # Legacy / synthetic path: Track identity is already locked to a specific student
            cand_map = {c["student_id"]: c for c in candidates}
            if state.locked_student_id in cand_map:
                locked_info = cand_map[state.locked_student_id]
                locked_sim = float(locked_info["similarity"])
                if locked_sim >= self.low_confidence_threshold:
                    state.vote_counts[state.locked_student_id] = (
                        state.vote_counts.get(state.locked_student_id, 0) + 1
                    )
                    state.similarity_history.setdefault(
                        state.locked_student_id, []
                    ).append(locked_sim)
                    c_prof = self.registry.get_student(state.locked_student_id)
                    mean_sim = float(
                        np.mean(state.similarity_history[state.locked_student_id])
                    )
                    decision = IdentityDecision(
                        stream_track_uid=state.stream_track_uid,
                        camera_id=state.camera_id,
                        session_id=state.session_id,
                        track_id=state.track_id,
                        status=STATUS_CONFIRMED,
                        student_id=state.locked_student_id,
                        name=c_prof.name if c_prof else state.locked_student_id,
                        roll_number=c_prof.roll_number if c_prof else None,
                        similarity=round(mean_sim, 4),
                        candidate_margin=round(candidate_margin, 4),
                        candidate_student_id=state.locked_student_id,
                        candidate_name=c_prof.name if c_prof else None,
                        observations=state.observations,
                        confirmations=state.vote_counts[state.locked_student_id],
                        frame_id=frame_id,
                        inference_performed=True,
                        latency_ms=round(match_ms, 3),
                    )
                else:
                    c_prof = self.registry.get_student(state.locked_student_id)
                    decision = IdentityDecision(
                        stream_track_uid=state.stream_track_uid,
                        camera_id=state.camera_id,
                        session_id=state.session_id,
                        track_id=state.track_id,
                        status=STATUS_LOW_CONFIDENCE,
                        student_id=None,
                        name=STATUS_LOW_CONFIDENCE,
                        roll_number=None,
                        similarity=round(locked_sim, 4),
                        candidate_margin=round(candidate_margin, 4),
                        candidate_student_id=state.locked_student_id,
                        candidate_name=c_prof.name if c_prof else None,
                        observations=state.observations,
                        confirmations=state.vote_counts.get(
                            state.locked_student_id, 0
                        ),
                        frame_id=frame_id,
                        inference_performed=True,
                        latency_ms=round(match_ms, 3),
                    )
            else:
                decision = IdentityDecision(
                    stream_track_uid=state.stream_track_uid,
                    camera_id=state.camera_id,
                    session_id=state.session_id,
                    track_id=state.track_id,
                    status=STATUS_UNKNOWN,
                    student_id=None,
                    name=STATUS_UNKNOWN,
                    roll_number=None,
                    similarity=round(top_sim, 4),
                    candidate_margin=round(candidate_margin, 4),
                    candidate_student_id=None,
                    candidate_name=None,
                    observations=state.observations,
                    confirmations=0,
                    frame_id=frame_id,
                    inference_performed=True,
                    latency_ms=round(match_ms, 3),
                )
        elif top_sim >= self.confirm_threshold and candidate_margin >= self.min_margin:
            state.vote_counts[top_sid] = state.vote_counts.get(top_sid, 0) + 1
            state.similarity_history.setdefault(top_sid, []).append(top_sim)

            best_sid = max(
                state.vote_counts.keys(),
                key=lambda s: (state.vote_counts[s], np.mean(state.similarity_history[s])),
            )
            best_votes = state.vote_counts[best_sid]
            mean_sim = float(np.mean(state.similarity_history[best_sid]))
            best_profile = self.registry.get_student(best_sid)

            if best_votes >= self.min_confirmations and best_profile is not None:
                state.confirmed_student_id = best_sid
                state.locked_student_id = best_sid
                decision = IdentityDecision(
                    stream_track_uid=state.stream_track_uid,
                    camera_id=state.camera_id,
                    session_id=state.session_id,
                    track_id=state.track_id,
                    status=STATUS_CONFIRMED,
                    student_id=best_sid,
                    name=best_profile.name,
                    roll_number=best_profile.roll_number,
                    similarity=round(mean_sim, 4),
                    candidate_margin=round(candidate_margin, 4),
                    candidate_student_id=best_sid,
                    candidate_name=best_profile.name,
                    observations=state.observations,
                    confirmations=best_votes,
                    frame_id=frame_id,
                    inference_performed=True,
                    latency_ms=round(match_ms, 3),
                )
            else:
                decision = IdentityDecision(
                    stream_track_uid=state.stream_track_uid,
                    camera_id=state.camera_id,
                    session_id=state.session_id,
                    track_id=state.track_id,
                    status=STATUS_TENTATIVE,
                    student_id=None,
                    name=STATUS_TENTATIVE,
                    roll_number=None,
                    similarity=round(top_sim, 4),
                    candidate_margin=round(candidate_margin, 4),
                    candidate_student_id=top_sid,
                    candidate_name=top_name,
                    observations=state.observations,
                    confirmations=best_votes,
                    frame_id=frame_id,
                    inference_performed=True,
                    latency_ms=round(match_ms, 3),
                )
        elif top_sim >= self.low_confidence_threshold:
            decision = IdentityDecision(
                stream_track_uid=state.stream_track_uid,
                camera_id=state.camera_id,
                session_id=state.session_id,
                track_id=state.track_id,
                status=STATUS_LOW_CONFIDENCE,
                student_id=None,
                name=STATUS_LOW_CONFIDENCE,
                roll_number=None,
                similarity=round(top_sim, 4),
                candidate_margin=round(candidate_margin, 4),
                candidate_student_id=top_sid,
                candidate_name=top_name,
                observations=state.observations,
                confirmations=state.vote_counts.get(top_sid, 0),
                frame_id=frame_id,
                inference_performed=True,
                latency_ms=round(match_ms, 3),
            )
        else:
            decision = IdentityDecision(
                stream_track_uid=state.stream_track_uid,
                camera_id=state.camera_id,
                session_id=state.session_id,
                track_id=state.track_id,
                status=STATUS_UNKNOWN,
                student_id=None,
                name=STATUS_UNKNOWN,
                roll_number=None,
                similarity=round(top_sim, 4),
                candidate_margin=round(candidate_margin, 4),
                candidate_student_id=None,
                candidate_name=None,
                observations=state.observations,
                confirmations=0,
                frame_id=frame_id,
                inference_performed=True,
                latency_ms=round(match_ms, 3),
            )

        state.last_decision = decision
        if resolve_conflicts:
            self._resolve_active_conflicts(
                camera_id=state.camera_id,
                session_id=state.session_id,
                current_frame_id=frame_id,
            )
        return state.last_decision

    def identify_crop(
        self,
        stream_track_uid: str,
        crop: np.ndarray,
        frame_id: int = 0,
        camera_id: Optional[str] = None,
        session_id: Optional[str] = None,
        track_id: Optional[int] = None,
        force_inference: bool = False,
        resolve_conflicts: bool = True,
    ) -> IdentityDecision:
        """
        Evaluates a person crop for `stream_track_uid` with automatic frame-interval skipping.
        If `force_inference=False` and the frame interval has not elapsed, returns the cached
        decision with `inference_performed=False` without invoking OSNet.
        """
        state = self._get_or_create_state(stream_track_uid, camera_id, session_id, track_id)
        state.frames_seen += 1
        state.last_seen_frame_id = frame_id

        if not force_inference and not self._should_run_inference(state):
            self.total_skipped_frames += 1
            if state.last_decision is not None:
                cached = IdentityDecision(
                    **{
                        **state.last_decision.to_dict(),
                        "frame_id": frame_id,
                        "inference_performed": False,
                        "latency_ms": 0.0,
                    }
                )
                state.last_decision = cached
                return cached

        if crop is None or crop.size == 0:
            decision = IdentityDecision(
                stream_track_uid=state.stream_track_uid,
                camera_id=state.camera_id,
                session_id=state.session_id,
                track_id=state.track_id,
                status=STATUS_UNKNOWN,
                student_id=None,
                name=STATUS_UNKNOWN,
                frame_id=frame_id,
                inference_performed=False,
            )
            state.last_decision = decision
            return decision

        # Undo the increment here because evaluate_embedding increments frames_seen
        state.frames_seen -= 1
        t_emb0 = time.perf_counter()
        if getattr(self.registry, "_has_biometric_gallery", False) and hasattr(
            self.reid_model, "extract_biometric_fast"
        ):
            raw_emb = self.reid_model.extract_biometric_fast(crop)
        else:
            raw_emb = self.reid_model.extract_embedding(crop)
        emb_ms = (time.perf_counter() - t_emb0) * 1000.0
        self.total_inferences += 1
        self.embedding_latencies_ms.append(emb_ms)

        decision = self.evaluate_embedding(
            stream_track_uid=stream_track_uid,
            embedding=raw_emb,
            frame_id=frame_id,
            camera_id=state.camera_id,
            session_id=state.session_id,
            track_id=state.track_id,
            resolve_conflicts=resolve_conflicts,
        )
        decision.latency_ms = round(decision.latency_ms + emb_ms, 3)
        return decision

    def process_frame(
        self,
        packet: FramePacket,
        tracked_people: List[TrackedPerson],
    ) -> List[IdentityDecision]:
        """
        PipelineController-compatible subscriber method (`callback(packet, tracked_people)`).
        Extracts person crops for active tracks, performs periodic OSNet inference,
        stabilizes identities across frames, and resolves simultaneous duplicate identity claims.
        """
        if packet is None or packet.frame is None or not tracked_people:
            return []

        decisions: List[IdentityDecision] = []
        for person in tracked_people:
            state = self._get_or_create_state(
                stream_track_uid=person.stream_track_uid,
                camera_id=person.camera_id,
                session_id=person.session_id,
                track_id=person.track_id,
            )

            # Check if we need to extract crop & run OSNet on this frame
            # Note: state.frames_seen will be incremented inside identify_crop
            next_frames_seen = state.frames_seen + 1
            steps_since_last = next_frames_seen - state.last_inference_step
            need_inference = (
                state.observations == 0
                or (
                    state.confirmed_student_id is not None
                    and steps_since_last >= self.reverify_interval_frames
                )
                or (
                    state.confirmed_student_id is None
                    and steps_since_last >= self.inference_interval_frames
                )
            )

            crop = person.extract_crop(packet.frame) if need_inference else None
            dec = self.identify_crop(
                stream_track_uid=person.stream_track_uid,
                crop=crop if crop is not None else np.zeros((32, 16, 3), dtype=np.uint8),
                frame_id=packet.frame_id,
                camera_id=person.camera_id,
                session_id=person.session_id,
                track_id=person.track_id,
                force_inference=False,
                resolve_conflicts=False,
            )
            decisions.append(dec)

        # Resolve conflicts across all simultaneous active tracks in this (camera_id, session_id)
        self._resolve_active_conflicts(
            camera_id=packet.camera_id,
            session_id=packet.session_id,
            current_frame_id=packet.frame_id,
            active_uids={p.stream_track_uid for p in tracked_people},
        )

        # Return updated decisions after conflict resolution
        return [
            self._track_states[p.stream_track_uid].last_decision
            for p in tracked_people
            if self._track_states.get(p.stream_track_uid) and self._track_states[p.stream_track_uid].last_decision
        ]

    def _resolve_active_conflicts(
        self,
        camera_id: str,
        session_id: str,
        current_frame_id: int,
        active_uids: Optional[set] = None,
    ) -> None:
        """
        Prevents two simultaneous active tracks within the same (camera_id, session_id)
        from claiming the same confirmed student_id.
        Assigns the identity to the track with stronger evidence (confirmations, similarity)
        and demotes competing tracks to LOW_CONFIDENCE.
        """
        # Collect active states in the same camera & session
        session_states: List[_TrackIdentityState] = []
        for uid, st in self._track_states.items():
            if st.camera_id != camera_id or st.session_id != session_id:
                continue
            if active_uids is not None:
                if uid not in active_uids:
                    continue
            else:
                if abs(current_frame_id - st.last_seen_frame_id) > self.active_track_ttl_frames:
                    continue
            if st.last_decision is not None and st.last_decision.status == STATUS_CONFIRMED and st.last_decision.student_id:
                session_states.append(st)

        # Group by claimed student_id
        by_student: Dict[str, List[_TrackIdentityState]] = {}
        for st in session_states:
            sid = st.last_decision.student_id
            if sid:
                by_student.setdefault(sid, []).append(st)

        for sid, claimants in by_student.items():
            if len(claimants) <= 1:
                continue

            # Rank claimants by (confirmations, similarity) descending
            claimants.sort(
                key=lambda s: (
                    s.last_decision.confirmations if s.last_decision else 0,
                    s.last_decision.similarity if s.last_decision else 0.0,
                ),
                reverse=True,
            )

            # Winner retains CONFIRMED status
            winner = claimants[0]
            winner.last_decision.conflict_demoted = False

            # Losers are demoted so two simultaneous tracks never claim the same student
            for loser in claimants[1:]:
                loser.confirmed_student_id = None
                loser.locked_student_id = None
                if loser.last_decision is not None:
                    loser.last_decision = IdentityDecision(
                        stream_track_uid=loser.stream_track_uid,
                        camera_id=loser.camera_id,
                        session_id=loser.session_id,
                        track_id=loser.track_id,
                        status=STATUS_LOW_CONFIDENCE,
                        student_id=None,
                        name=STATUS_LOW_CONFIDENCE,
                        roll_number=None,
                        similarity=loser.last_decision.similarity,
                        candidate_margin=loser.last_decision.candidate_margin,
                        candidate_student_id=sid,
                        candidate_name=loser.last_decision.candidate_name,
                        observations=loser.observations,
                        confirmations=loser.last_decision.confirmations,
                        frame_id=loser.last_decision.frame_id,
                        inference_performed=loser.last_decision.inference_performed,
                        conflict_demoted=True,
                        latency_ms=loser.last_decision.latency_ms,
                    )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Returns OSNet embedding and gallery matching latency metrics."""
        avg_emb = float(np.mean(self.embedding_latencies_ms)) if self.embedding_latencies_ms else 0.0
        avg_match = float(np.mean(self.matching_latencies_ms)) if self.matching_latencies_ms else 0.0
        return {
            "total_inferences": self.total_inferences,
            "total_skipped_frames": self.total_skipped_frames,
            "avg_embedding_latency_ms": round(avg_emb, 3),
            "avg_matching_latency_ms": round(avg_match, 3),
            "active_tracks_tracked": len(self._track_states),
        }
