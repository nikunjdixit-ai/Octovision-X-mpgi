from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("cctv_ai.reid.gallery")


def parse_stream_track_uid(identifier: str) -> Dict[str, Any]:
    """
    Parses camera_id, session_id, and track_id from namespaced UID.
    Handles formats:
      - 'CAM_01_SESS_01_1' -> camera_id='CAM_01', session_id='SESS_01', track_id=1
      - 'CAM_MALL_NORTH_SESS_123_42' -> camera_id='CAM_MALL_NORTH', session_id='SESS_123', track_id=42
      - 'CAM_MALL_01_1789899395_2' -> camera_id='CAM_MALL_01', session_id='1789899395', track_id=2
      - 'ID_1' -> track_id=1, camera_id=None, session_id=None
      - '1' -> track_id=1
    """
    meta: Dict[str, Any] = {
        "stream_track_uid": identifier,
        "camera_id": None,
        "session_id": None,
        "track_id": None,
    }

    # Match format: <cam>_(SESS_<sess>)_<track_id>
    sess_match = re.match(r"^(.*)_(SESS_[^_]+)_(\d+)$", identifier)
    if sess_match:
        meta["camera_id"] = sess_match.group(1)
        meta["session_id"] = sess_match.group(2)
        meta["track_id"] = int(sess_match.group(3))
        return meta

    # Match format: <cam>_<sess>_<track_id>
    generic_match = re.match(r"^(.*)_([^_]+)_(\d+)$", identifier)
    if generic_match:
        meta["camera_id"] = generic_match.group(1)
        meta["session_id"] = generic_match.group(2)
        meta["track_id"] = int(generic_match.group(3))
        return meta

    # Match format: ID_<track_id>
    if identifier.startswith("ID_") and identifier[3:].isdigit():
        meta["track_id"] = int(identifier[3:])
    elif identifier.isdigit():
        meta["track_id"] = int(identifier)

    return meta


class ReIDGallery:
    """
    High-performance in-memory gallery for person Re-ID candidate matching.
    Loads and normalizes embeddings into a contiguous 2D float32 matrix,
    allowing sub-millisecond vectorized similarity search.
    """

    def __init__(
        self,
        embeddings_dir: Optional[Union[str, Path]] = None,
        metadata_dir: Optional[Union[str, Path]] = None,
        expected_dim: int = 512,
    ):
        self.expected_dim = expected_dim
        self.entries: List[Dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None  # Shape: (N, expected_dim)

        if embeddings_dir is not None:
            self.load_from_directory(embeddings_dir, metadata_dir)

    def __len__(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalizes an embedding vector with zero-division safeguard."""
        v = np.asarray(vec, dtype=np.float32).flatten()
        if v.shape[0] != self.expected_dim:
            raise ValueError(
                f"Invalid embedding dimension {v.shape[0]}. Expected {self.expected_dim}."
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("Embedding contains non-finite values (NaN or Inf).")

        norm = float(np.linalg.norm(v))
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError("Embedding vector has zero or invalid norm.")

        return v / norm

    def add_entry(
        self,
        embedding: np.ndarray,
        stream_track_uid: str,
        camera_id: Optional[str] = None,
        session_id: Optional[str] = None,
        track_id: Optional[Union[int, str]] = None,
        frame_id: Optional[int] = None,
        timestamp: Optional[str] = None,
        source_video: Optional[str] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Adds a single person identity embedding into the in-memory gallery.
        """
        norm_v = self._normalize_vector(embedding)
        parsed = parse_stream_track_uid(stream_track_uid)

        entry = {
            "stream_track_uid": stream_track_uid,
            "camera_id": camera_id or parsed.get("camera_id"),
            "session_id": session_id or parsed.get("session_id"),
            "track_id": track_id if track_id is not None else parsed.get("track_id"),
            "frame_id": frame_id,
            "timestamp": timestamp,
            "source_video": source_video,
            "bbox": bbox,
            "embedding": norm_v,
            "metadata": metadata or {},
        }

        self.entries.append(entry)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Reconstructs the continuous 2D numpy matrix from entries list."""
        if not self.entries:
            self.matrix = None
            return
        vectors = [e["embedding"] for e in self.entries]
        self.matrix = np.vstack(vectors).astype(np.float32)

    def load_from_directory(
        self,
        embeddings_dir: Union[str, Path],
        metadata_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Loads all .npy embeddings and metadata files from directory.
        """
        emb_path = Path(embeddings_dir)
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {emb_path}")

        npy_files = sorted(emb_path.glob("*.npy"))
        if not npy_files:
            logger.info("No .npy embedding files found in %s", emb_path)
            self.entries = []
            self.matrix = None
            return

        # Preload metadata CSVs if available
        tracking_metadata: Dict[str, Dict[str, Any]] = {}
        if metadata_dir is not None:
            m_path = Path(metadata_dir)
            if m_path.exists():
                tracking_metadata = self._index_tracking_metadata(m_path)

        for npy_file in npy_files:
            uid = npy_file.stem
            try:
                raw_emb = np.load(npy_file)
                norm_emb = self._normalize_vector(raw_emb)
            except Exception as e:
                logger.warning("Skipping unreadable/invalid embedding %s: %s", npy_file.name, e)
                continue

            # Check for JSON metadata
            json_file = emb_path / f"{uid}.json"
            json_meta: Dict[str, Any] = {}
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as fh:
                        json_meta = json.load(fh)
                except Exception as e:
                    logger.warning("Could not read metadata JSON %s: %s", json_file.name, e)

            parsed = parse_stream_track_uid(uid)
            track_meta = tracking_metadata.get(uid, {})

            entry = {
                "stream_track_uid": uid,
                "camera_id": json_meta.get("camera_id", parsed.get("camera_id")),
                "session_id": json_meta.get("session_id", parsed.get("session_id")),
                "track_id": json_meta.get(
                    "track_id",
                    track_meta.get("track_id", parsed.get("track_id")),
                ),
                "frame_id": json_meta.get("frame_id", track_meta.get("frame_id")),
                "timestamp": json_meta.get("timestamp", track_meta.get("timestamp")),
                "source_video": json_meta.get(
                    "source_video", track_meta.get("source_video")
                ),
                "bbox": json_meta.get("bbox", track_meta.get("bbox")),
                "embedding": norm_emb,
                "metadata": {**json_meta, **track_meta},
            }
            self.entries.append(entry)

        self._rebuild_matrix()
        logger.info(
            "Loaded %d gallery identities from %s (matrix shape: %s)",
            len(self.entries),
            emb_path,
            self.matrix.shape if self.matrix is not None else None,
        )

    def _index_tracking_metadata(self, metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Scans tracking metadata CSVs and indexes by track identity."""
        index: Dict[str, Dict[str, Any]] = {}
        csv_files = sorted(metadata_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            if "track_id" not in df.columns:
                continue

            if "frame" in df.columns:
                first_rows = df.sort_values("frame").drop_duplicates(subset=["track_id"], keep="first")
            else:
                first_rows = df.drop_duplicates(subset=["track_id"], keep="first")

            for _, row in first_rows.iterrows():
                track_id = row["track_id"]
                video_name = str(row.get("video_name", ""))
                frame = int(row["frame"]) if "frame" in row and not pd.isna(row["frame"]) else None
                bbox = None
                if all(c in row and not pd.isna(row[c]) for c in ("x1", "y1", "x2", "y2")):
                    bbox = (
                        int(row["x1"]),
                        int(row["y1"]),
                        int(row["x2"]),
                        int(row["y2"]),
                    )

                meta_entry = {
                    "source_video": video_name,
                    "frame_id": frame,
                    "bbox": bbox,
                    "track_id": int(track_id) if str(track_id).isdigit() else str(track_id),
                }

                # Index by raw track_id string and ID_{track_id}
                index[str(track_id)] = meta_entry
                index[f"ID_{track_id}"] = meta_entry

        return index

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """
        Executes vectorized in-memory cosine similarity search against all gallery identities.

        Args:
            query_embedding: 512-dim embedding vector.
            top_k: Maximum number of top matches to return.
            threshold: Minimum cosine similarity score required (0.0 to 1.0).

        Returns:
            List of match dictionaries sorted by descending similarity.
            Returns empty list if no candidate meets the threshold or gallery is empty.
        """
        if self.is_empty() or self.matrix is None:
            return []

        q = self._normalize_vector(query_embedding)

        # Vectorized dot product (cosine similarity since both are L2-normalized)
        similarities = np.dot(self.matrix, q)

        # Find candidates meeting threshold
        qualifying_mask = similarities >= threshold
        if not np.any(qualifying_mask):
            return []

        qualifying_indices = np.where(qualifying_mask)[0]

        # Sort descending by similarity
        sorted_indices = qualifying_indices[
            np.argsort(-similarities[qualifying_indices])
        ]

        top_indices = sorted_indices[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            entry = self.entries[idx]
            sim = float(similarities[idx])
            match_data = {
                "rank": rank,
                "similarity": round(sim, 4),
                "stream_track_uid": entry["stream_track_uid"],
                "camera_id": entry.get("camera_id"),
                "session_id": entry.get("session_id"),
                "track_id": entry.get("track_id"),
                "frame_id": entry.get("frame_id"),
                "timestamp": entry.get("timestamp"),
                "source_video": entry.get("source_video"),
                "bbox": entry.get("bbox"),
                "metadata": entry.get("metadata", {}),
            }
            results.append(match_data)

        return results
