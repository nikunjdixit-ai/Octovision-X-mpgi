from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.reid.reid_model import ReIDModel

logger = logging.getLogger("trace.identity.enrollment")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_EMBEDDING_DIM = 512


class EnrollmentError(ValueError):
    """Raised when student enrollment validation fails due to invalid data, corrupt images, or duplicates."""
    pass


class EmbeddingExtractorProtocol(Protocol):
    """Protocol for OSNet embedding extraction, fulfilled by src.reid.reid_model.ReIDModel."""
    def extract_embedding(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        ...


def sanitize_roll_number(value: Any) -> str:
    """
    Normalizes roll_number / roll_no into an exact string of digits/characters,
    preventing float conversion ('2400461520039.0') or scientific notation ('2.40046e+12').
    """
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isfinite(value) and value.is_integer():
            return str(int(round(value)))
        return str(value)
    s = str(value).strip()
    if not s:
        return ""
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1].strip()
    elif s.startswith("'"):
        s = s[1:].strip()
    if re.match(r"^\d+\.0+$", s):
        s = s.split(".")[0]
    elif re.match(r"^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$", s):
        try:
            from decimal import Decimal
            dec = Decimal(s)
            if dec == int(dec):
                s = str(int(dec))
        except Exception:
            pass
    return str(s)


@dataclass
class StudentProfile:
    """
    Metadata and enrollment profile for an enrolled student in TRACE.
    """
    student_id: str
    name: str
    roll_number: str
    department: str = "General"
    section: str = "A"
    num_samples: int = 0
    enrolled_at: str = ""
    source_images: List[Dict[str, Any]] = field(default_factory=list)
    enrollment_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.student_id = str(self.student_id).strip()
        self.name = str(self.name).strip()
        self.roll_number = sanitize_roll_number(self.roll_number)
        self.department = str(self.department).strip()
        self.section = str(self.section).strip()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["roll_number"] = str(self.roll_number)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StudentProfile:
        raw_roll = data.get("roll_number") if "roll_number" in data else data.get("roll_no", "")
        return cls(
            student_id=str(data["student_id"]),
            name=str(data["name"]),
            roll_number=sanitize_roll_number(raw_roll),
            department=str(data.get("department", "General")),
            section=str(data.get("section", "A")),
            num_samples=int(data.get("num_samples", 0)),
            enrolled_at=str(data.get("enrolled_at", "")),
            source_images=list(data.get("source_images", [])),
            enrollment_metadata=dict(data.get("enrollment_metadata", {})),
        )


def normalize_l2(vector: np.ndarray, expected_dim: int = EXPECTED_EMBEDDING_DIM) -> np.ndarray:
    """
    Validates dimensionality, checks finiteness, converts to float32,
    and returns a unit L2-normalized vector.
    """
    v = np.asarray(vector, dtype=np.float32).flatten()
    if v.shape[0] != expected_dim:
        raise EnrollmentError(
            f"Invalid embedding dimension {v.shape[0]}; expected {expected_dim}."
        )
    if not np.all(np.isfinite(v)):
        raise EnrollmentError("Embedding contains non-finite values (NaN or Inf).")

    norm = float(np.linalg.norm(v))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise EnrollmentError("Embedding vector has zero or invalid L2 norm.")

    return (v / norm).astype(np.float32)


def parse_student_folder_name(folder_name: str) -> Tuple[str, str, str]:
    """
    Parses a folder name such as '01_Rahul_Sharma' into:
      (student_id, roll_number, name)
    Examples:
      '01_Rahul_Sharma' -> ('01_Rahul_Sharma', '01', 'Rahul Sharma')
      'STU102_Aman_Verma' -> ('STU102_Aman_Verma', 'STU102', 'Aman Verma')
    Raises EnrollmentError if folder_name is empty or invalid.
    """
    cleaned = folder_name.strip()
    if not cleaned:
        raise EnrollmentError("Student folder name cannot be empty.")

    # Handle generic indexed folders like 'student_1', 'student_2', 'person_01'
    indexed_match = re.match(r"^([A-Za-z]+)_(\d+)$", cleaned)
    if indexed_match:
        prefix = indexed_match.group(1).strip().capitalize()
        idx_str = indexed_match.group(2).strip()
        return cleaned, cleaned, f"{prefix} {idx_str}"

    match = re.match(r"^([A-Za-z0-9-]+)_(.+)$", cleaned)
    if match:
        roll_number = match.group(1).strip()
        raw_name = match.group(2).replace("_", " ").strip()
        if not raw_name:
            raise EnrollmentError(f"Missing student name in folder '{folder_name}'.")
        return cleaned, roll_number, raw_name

    # Fallback if no underscore separator is used
    return cleaned, cleaned, cleaned.replace("_", " ")


def validate_and_load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Strictly validates an enrollment image file:
    1. File exists and is a regular file.
    2. Extension is in SUPPORTED_IMAGE_EXTENSIONS.
    3. File is non-empty and passes PIL verification + OpenCV BGR decode.
    Returns (bgr_image_ndarray, source_image_metadata).
    """
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise EnrollmentError(f"Image file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise EnrollmentError(
            f"Unsupported image extension '{ext}' for file '{path.name}'. "
            f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise EnrollmentError(f"Image file is empty (0 bytes): {path}")

    # 1. Verify header & integrity via PIL
    try:
        with Image.open(path) as pil_img:
            pil_img.verify()
    except Exception as exc:
        raise EnrollmentError(f"Corrupted or invalid image file '{path.name}': {exc}") from exc

    # 2. Decode pixel array via OpenCV
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise EnrollmentError(f"Unreadable image pixels in file '{path.name}'.")

    h, w = bgr.shape[:2]
    if h < 16 or w < 8:
        raise EnrollmentError(
            f"Image '{path.name}' resolution ({w}x{h}) is too small for OSNet person Re-ID."
        )

    sha256_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    meta = {
        "filename": path.name,
        "width": int(w),
        "height": int(h),
        "size_bytes": int(file_size),
        "sha256_prefix": sha256_hash,
    }
    return bgr, meta


class StudentRegistry:
    """
    Manages enrolled student profiles, multi-sample 512-D L2-normalized OSNet embeddings,
    normalized student prototype embeddings, local persistence, and fast vectorized lookup.
    """

    def __init__(
        self,
        registry_dir: Optional[Union[str, Path]] = None,
        reid_model: Optional[EmbeddingExtractorProtocol] = None,
        expected_dim: int = EXPECTED_EMBEDDING_DIM,
    ):
        self.expected_dim = expected_dim
        self.registry_dir = Path(registry_dir) if registry_dir is not None else None
        self._reid_model = reid_model

        self.profiles: Dict[str, StudentProfile] = {}
        self.roll_to_student_id: Dict[str, str] = {}
        self.sample_embeddings: Dict[str, np.ndarray] = {}     # student_id -> (K, 512) float32
        self.prototype_embeddings: Dict[str, np.ndarray] = {}  # student_id -> (512,) float32

        # Contiguous matrices for vectorized search
        self._student_ids_ordered: List[str] = []
        self._prototype_matrix: Optional[np.ndarray] = None    # (M, 512)
        self._all_samples_matrix: Optional[np.ndarray] = None  # (Total_Samples, 512)
        self._sample_owner_ids: List[str] = []

        if self.registry_dir is not None and (self.registry_dir / "students_registry.json").exists():
            self.load(self.registry_dir)

    @property
    def reid_model(self) -> EmbeddingExtractorProtocol:
        """Lazily initializes the shared OSNet ReIDModel if none was injected."""
        if self._reid_model is None:
            self._reid_model = ReIDModel()
        return self._reid_model

    def __len__(self) -> int:
        return len(self.profiles)

    def is_empty(self) -> bool:
        return len(self.profiles) == 0

    def list_students(self) -> List[StudentProfile]:
        return [self.profiles[sid] for sid in self._student_ids_ordered]

    def get_student(self, student_id: str) -> Optional[StudentProfile]:
        return self.profiles.get(student_id)

    def get_prototype(self, student_id: str) -> Optional[np.ndarray]:
        return self.prototype_embeddings.get(student_id)

    def get_samples(self, student_id: str) -> Optional[np.ndarray]:
        return self.sample_embeddings.get(student_id)

    def _rebuild_indices(self) -> None:
        from src.reid.reid_model import unpack_hybrid_embedding

        self._student_ids_ordered = sorted(self.profiles.keys())
        self._has_biometric_gallery = False
        self._biometric_gallery: Dict[str, Dict[str, Any]] = {}
        self._mean_hd = np.zeros(128, dtype=np.float32)
        self._mean_bd = np.zeros(128, dtype=np.float32)

        if not self._student_ids_ordered:
            self._prototype_matrix = None
            self._all_samples_matrix = None
            self._sample_owner_ids = []
            return

        prototypes = []
        all_samples = []
        sample_owners = []

        for sid in self._student_ids_ordered:
            prototypes.append(self.prototype_embeddings[sid])
            samples = self.sample_embeddings[sid]
            for row in samples:
                all_samples.append(row)
                sample_owners.append(sid)

        self._prototype_matrix = np.vstack(prototypes).astype(np.float32)
        self._all_samples_matrix = np.vstack(all_samples).astype(np.float32)
        self._sample_owner_ids = sample_owners

        # Check if gallery uses hybrid biometric embeddings
        unpacked_by_sid: Dict[str, List[Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]]] = {}
        bio_count = 0
        all_hd_list = []
        all_bd_list = []
        for sid in self._student_ids_ordered:
            items = [unpack_hybrid_embedding(row) for row in self.sample_embeddings[sid]]
            unpacked_by_sid[sid] = items
            for is_b, sf, hd, bd, q in items:
                if is_b:
                    bio_count += 1
                all_hd_list.append(hd)
                all_bd_list.append(bd)

        self._pose_basis: Optional[np.ndarray] = None
        if bio_count >= max(2, len(all_samples) // 2) and len(self._student_ids_ordered) >= 2:
            self._has_biometric_gallery = True
            self._mean_hd = np.mean(np.array(all_hd_list, dtype=np.float32), axis=0)
            self._mean_bd = np.mean(np.array(all_bd_list, dtype=np.float32), axis=0)

            pose_diffs: List[np.ndarray] = []
            temp_bio: Dict[str, Dict[str, Any]] = {}

            for sid, items in unpacked_by_sid.items():
                sfs = [it[1] for it in items]
                qs = [it[4] for it in items]
                best_i = int(np.argmax(qs))
                f0_sf = sfs[best_i]

                peer_sims = []
                for i in range(len(sfs)):
                    others = [float(np.dot(sfs[i], sfs[j])) for j in range(len(sfs)) if j != i]
                    peer_sims.append(max(others) if others else 1.0)

                valid_sfs = []
                for i in range(len(sfs)):
                    if not (len(items) >= 3 and peer_sims[i] < 0.25):
                        valid_sfs.append(sfs[i])
                        if i != best_i:
                            diff = sfs[i] - f0_sf
                            if float(np.linalg.norm(diff)) > 0.10:
                                pose_diffs.append(diff)
                if not valid_sfs:
                    valid_sfs = [f0_sf]

                vecs = [
                    self._build_centered_vec(it[1], it[2], it[3], it[0])
                    for it in items
                ]
                weights = [
                    0.01 if (len(items) >= 3 and peer_sims[i] < 0.25) else 1.0
                    for i in range(len(items))
                ]
                w_arr = np.array(weights, dtype=np.float32)
                w_arr /= np.sum(w_arr)
                proto = np.sum(np.array(vecs) * w_arr[:, None], axis=0)
                norm_p = float(np.linalg.norm(proto))
                if norm_p > 1e-8:
                    proto = (proto / norm_p).astype(np.float32)
                valid_samples = [
                    vecs[i]
                    for i in range(len(vecs))
                    if not (len(items) >= 3 and peer_sims[i] < 0.25)
                ]
                temp_bio[sid] = {
                    "proto": proto,
                    "samples": valid_samples if valid_samples else vecs,
                    "frontal_sf": f0_sf,
                    "valid_sfs": valid_sfs,
                }

            if len(pose_diffs) >= 4:
                d_pose = np.vstack(pose_diffs).astype(np.float32)
                _, _, vt = np.linalg.svd(d_pose, full_matrices=False)
                k_basis = min(4, vt.shape[0])
                self._pose_basis = vt[:k_basis].astype(np.float32)

            for sid, g_item in temp_bio.items():
                f0_dep = self._depose_sf(g_item["frontal_sf"])
                val_deps = [self._depose_sf(s) for s in g_item["valid_sfs"]]
                dep_proto = 0.55 * f0_dep + 0.45 * np.mean(val_deps, axis=0)
                n_dep = float(np.linalg.norm(dep_proto))
                g_item["dep_proto_sf"] = (
                    (dep_proto / n_dep).astype(np.float32) if n_dep > 1e-8 else f0_dep
                )
                self._biometric_gallery[sid] = g_item

    def _depose_sf(self, sf_vec: np.ndarray) -> np.ndarray:
        """Projects out the within-class frontal-to-side pose subspace from a 128-D SFace vector."""
        if getattr(self, "_pose_basis", None) is None:
            return sf_vec
        vp = sf_vec - 0.75 * (self._pose_basis.T @ (self._pose_basis @ sf_vec))
        n = float(np.linalg.norm(vp))
        return (vp / n).astype(np.float32) if n > 1e-8 else sf_vec

    def _build_centered_vec(
        self,
        sf: np.ndarray,
        hd: np.ndarray,
        bd: np.ndarray,
        has_face: bool,
        w_face: float = 0.75,
        w_head: float = 0.15,
        w_body: float = 0.10,
    ) -> np.ndarray:
        hd_c = hd - 0.80 * self._mean_hd
        n_hd = float(np.linalg.norm(hd_c))
        hd_c = (hd_c / n_hd).astype(np.float32) if n_hd > 1e-8 else hd

        bd_c = bd - 0.80 * self._mean_bd
        n_bd = float(np.linalg.norm(bd_c))
        bd_c = (bd_c / n_bd).astype(np.float32) if n_bd > 1e-8 else bd

        if has_face and float(np.linalg.norm(sf)) > 1e-6:
            wf_half = float(np.sqrt(w_face / 2.0))
            wh = float(np.sqrt(w_head))
            wb = float(np.sqrt(w_body))
            v = np.concatenate([wf_half * sf, wf_half * sf, wh * hd_c, wb * bd_c]).astype(np.float32)
        else:
            v = np.concatenate([
                np.zeros(256, dtype=np.float32),
                float(np.sqrt(0.55)) * hd_c,
                float(np.sqrt(0.45)) * bd_c,
            ]).astype(np.float32)
        nv = float(np.linalg.norm(v))
        return (v / nv).astype(np.float32) if nv > 1e-8 else v

    @staticmethod
    def _calibrate_biometric_score(raw_sim: float) -> float:
        x = float(np.clip(raw_sim, 0.0, 1.0))
        if x <= 0.300:
            return (0.60 / 0.300) * x
        elif x <= 0.330:
            return 0.60 + ((0.73 - 0.60) / (0.330 - 0.300)) * (x - 0.300)
        else:
            return 0.73 + ((1.00 - 0.73) / (1.00 - 0.330)) * (x - 0.330)

    def enroll_student(
        self,
        student_id: str,
        name: str,
        roll_number: str,
        image_paths: List[Union[str, Path]],
        department: str = "General",
        section: str = "A",
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> StudentProfile:
        """
        Enrolls a single student from one or more reference image paths.
        Validates against duplicates, missing/empty image lists, unsupported formats, and corrupt images.
        """
        student_id = student_id.strip()
        name = name.strip()
        roll_number = roll_number.strip()

        if not student_id or not name or not roll_number:
            raise EnrollmentError("student_id, name, and roll_number must all be non-empty.")

        if student_id in self.profiles:
            raise EnrollmentError(f"Duplicate student_id '{student_id}' is already enrolled.")

        if roll_number in self.roll_to_student_id:
            existing_sid = self.roll_to_student_id[roll_number]
            raise EnrollmentError(
                f"Duplicate roll_number '{roll_number}' is already enrolled under '{existing_sid}'."
            )

        if not image_paths:
            raise EnrollmentError(f"No reference images provided for student '{student_id}'.")

        extracted_vectors: List[np.ndarray] = []
        source_meta_list: List[Dict[str, Any]] = []

        for img_path in image_paths:
            bgr_img, img_meta = validate_and_load_image(img_path)
            raw_emb = self.reid_model.extract_embedding(bgr_img)
            norm_emb = normalize_l2(raw_emb, expected_dim=self.expected_dim)
            extracted_vectors.append(norm_emb)
            source_meta_list.append(img_meta)

        samples_matrix = np.vstack(extracted_vectors).astype(np.float32)
        # Compute normalized student prototype (mean of normalized embeddings, re-normalized to unit L2)
        mean_vec = np.mean(samples_matrix, axis=0)
        prototype_vec = normalize_l2(mean_vec, expected_dim=self.expected_dim)

        profile = StudentProfile(
            student_id=student_id,
            name=name,
            roll_number=roll_number,
            department=department,
            section=section,
            num_samples=len(extracted_vectors),
            enrolled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_images=source_meta_list,
            enrollment_metadata=metadata or {},
        )

        self.profiles[student_id] = profile
        self.roll_to_student_id[roll_number] = student_id
        self.sample_embeddings[student_id] = samples_matrix
        self.prototype_embeddings[student_id] = prototype_vec
        self._rebuild_indices()

        if persist and self.registry_dir is not None:
            self.save(self.registry_dir)

        logger.info(
            "Enrolled student '%s' (%s, Roll: %s) with %d OSNet embeddings.",
            name,
            student_id,
            roll_number,
            len(extracted_vectors),
        )
        return profile

    def enroll_from_directory(
        self,
        students_dir: Union[str, Path],
        department: str = "General",
        section: str = "A",
        persist: bool = True,
        metadata_csv: Optional[Union[str, Path]] = None,
    ) -> List[StudentProfile]:
        """
        Enrolls all student subfolders from a root directory:
        students/
        ├── 01_Rahul_Sharma/
        │   ├── 01.jpg
        │   ├── 02.jpg
        │   └── 03.jpg
        ├── 02_Aman_Verma/
        ...
        Optionally applies a student roster CSV (student_id,name,roll_no,department,section)
        if provided via `metadata_csv` or present as `metadata.csv` / `students_metadata.csv` in `students_dir`.
        """
        root = Path(students_dir)
        if not root.exists() or not root.is_dir():
            raise EnrollmentError(f"Students root directory does not exist: {root}")

        subdirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
        if not subdirs:
            raise EnrollmentError(f"No student subdirectories found in '{root}'.")

        staged_plans: List[Tuple[str, str, str, List[Path]]] = []
        seen_ids: set = set()
        seen_rolls: set = set()

        for folder in subdirs:
            student_id, roll_number, student_name = parse_student_folder_name(folder.name)

            if student_id in self.profiles or student_id in seen_ids:
                raise EnrollmentError(f"Duplicate student identity '{student_id}' detected.")
            if roll_number in self.roll_to_student_id or roll_number in seen_rolls:
                raise EnrollmentError(f"Duplicate roll_number '{roll_number}' detected in folder '{folder.name}'.")

            seen_ids.add(student_id)
            seen_rolls.add(roll_number)

            files = sorted([f for f in folder.iterdir() if not f.name.startswith(".")])
            if not files:
                raise EnrollmentError(f"Student folder '{folder.name}' is empty.")

            for f in files:
                if f.is_dir():
                    raise EnrollmentError(
                        f"Unexpected nested subdirectory '{f.name}' inside student folder '{folder.name}'."
                    )
                if f.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    raise EnrollmentError(
                        f"Unsupported file '{f.name}' in student folder '{folder.name}'. "
                        f"Only {sorted(SUPPORTED_IMAGE_EXTENSIONS)} are allowed."
                    )

            staged_plans.append((student_id, student_name, roll_number, files))

        enrolled_profiles: List[StudentProfile] = []
        for student_id, student_name, roll_number, img_files in staged_plans:
            prof = self.enroll_student(
                student_id=student_id,
                name=student_name,
                roll_number=roll_number,
                image_paths=img_files,
                department=department,
                section=section,
                metadata={"folder_name": student_id},
                persist=False,
            )
            enrolled_profiles.append(prof)

        # Check for explicit or auto-discovered metadata CSV
        csv_to_apply: Optional[Path] = Path(metadata_csv) if metadata_csv is not None else None
        if csv_to_apply is None:
            for candidate_name in ("students_metadata.csv", "metadata.csv", "students.csv"):
                cand = root / candidate_name
                if cand.exists() and cand.is_file():
                    csv_to_apply = cand
                    break

        if csv_to_apply is not None:
            self.apply_metadata_csv(csv_to_apply, persist=False)
            enrolled_profiles = self.list_students()

        if persist and self.registry_dir is not None:
            self.save(self.registry_dir)

        return enrolled_profiles

    def apply_metadata_csv(
        self,
        csv_path: Union[str, Path],
        persist: bool = True,
    ) -> int:
        """
        Applies authoritative student metadata from a CSV file with columns:
          student_id, name, roll_no (or roll_number), department, section
        Matches rows to enrolled StudentProfile entries by student_id, folder prefix (initial roll_number),
        or normalized student name, and validates uniqueness of roll numbers.
        """
        import csv as csv_module

        cpath = Path(csv_path)
        if not cpath.exists() or not cpath.is_file():
            raise EnrollmentError(f"Metadata CSV file not found: {cpath}")

        with open(cpath, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv_module.DictReader(fh)
            if not reader.fieldnames:
                raise EnrollmentError(f"Metadata CSV '{cpath.name}' has no header columns.")
            rows = list(reader)

        if not rows:
            raise EnrollmentError(f"Metadata CSV '{cpath.name}' contains no student rows.")

        updated_count = 0
        for row in rows:
            norm_row = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None and v is not None}
            csv_sid = norm_row.get("student_id", "")
            csv_name = norm_row.get("name", "")
            csv_roll = sanitize_roll_number(norm_row.get("roll_no") or norm_row.get("roll_number") or "")
            csv_dept = norm_row.get("department", "")
            csv_sec = norm_row.get("section", "")

            if not csv_roll and not csv_sid:
                continue

            # Find matching enrolled profile
            matched_sid: Optional[str] = None
            if csv_sid and csv_sid in self.profiles:
                matched_sid = csv_sid
            elif csv_sid and csv_sid in self.roll_to_student_id:
                matched_sid = self.roll_to_student_id[csv_sid]
            else:
                # Match by prefix or name
                for sid, prof in self.profiles.items():
                    folder_prefix = sid.split("_")[0]
                    if (csv_sid and folder_prefix == csv_sid) or (
                        csv_name and prof.name.lower() == csv_name.lower()
                    ):
                        matched_sid = sid
                        break

            if matched_sid is None:
                continue

            prof = self.profiles[matched_sid]
            if csv_name:
                prof.name = csv_name
            if csv_dept:
                prof.department = csv_dept
            if csv_sec:
                prof.section = csv_sec
            if csv_roll:
                # Check for duplicate roll number collision with a different student
                existing_owner = self.roll_to_student_id.get(csv_roll)
                if existing_owner is not None and existing_owner != matched_sid:
                    raise EnrollmentError(
                        f"Duplicate roll_number '{csv_roll}' in metadata CSV conflicts with '{existing_owner}'."
                    )
                # Remove old roll mapping
                if prof.roll_number in self.roll_to_student_id and self.roll_to_student_id[prof.roll_number] == matched_sid:
                    del self.roll_to_student_id[prof.roll_number]
                prof.roll_number = csv_roll
                self.roll_to_student_id[csv_roll] = matched_sid

            updated_count += 1

        self._rebuild_indices()
        if persist and self.registry_dir is not None:
            self.save(self.registry_dir)

        return updated_count

    def enroll_from_zip(
        self,
        zip_path: Union[str, Path],
        department: str = "General",
        section: str = "A",
        persist: bool = True,
        metadata_csv: Optional[Union[str, Path]] = None,
    ) -> List[StudentProfile]:
        """
        Validates and extracts a ZIP archive containing student reference folders,
        then enrolls all students.
        Supports either:
          - archive.zip/students/01_Rahul_Sharma/01.jpg
          - archive.zip/01_Rahul_Sharma/01.jpg
        """
        zpath = Path(zip_path)
        if not zpath.exists() or not zpath.is_file():
            raise EnrollmentError(f"Student ZIP file not found: {zpath}")

        if not zipfile.is_zipfile(zpath):
            raise EnrollmentError(f"Invalid or corrupted ZIP archive: {zpath}")

        temp_extract_dir = tempfile.mkdtemp(prefix="trace_enroll_zip_")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                members = zf.namelist()
                if not members:
                    raise EnrollmentError(f"ZIP archive is empty: {zpath}")

                for member in members:
                    member_path = Path(member)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise EnrollmentError(f"Unsafe path detected in ZIP archive: {member}")

                zf.extractall(temp_extract_dir)

            extracted_root = Path(temp_extract_dir)
            macosx_dir = extracted_root / "__MACOSX"
            if macosx_dir.exists():
                shutil.rmtree(macosx_dir, ignore_errors=True)

            top_dirs = [p for p in extracted_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
            if len(top_dirs) == 1 and top_dirs[0].name.lower() in {"students", "enrollment", "dataset"}:
                target_root = top_dirs[0]
            else:
                target_root = extracted_root

            # Check if metadata.csv is at extracted_root while folders are in target_root
            effective_csv = metadata_csv
            if effective_csv is None:
                for cname in ("students_metadata.csv", "metadata.csv", "students.csv"):
                    if (extracted_root / cname).exists():
                        effective_csv = extracted_root / cname
                        break

            return self.enroll_from_directory(
                students_dir=target_root,
                department=department,
                section=section,
                persist=persist,
                metadata_csv=effective_csv,
            )
        finally:
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

    def save(self, registry_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Persists student metadata JSON and L2-normalized float32 embeddings (.npy) to disk.
        """
        target_dir = Path(registry_dir) if registry_dir is not None else self.registry_dir
        if target_dir is None:
            raise EnrollmentError("No registry_dir configured for saving StudentRegistry.")

        self.registry_dir = target_dir
        emb_dir = target_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": "1.0",
            "embedding_backbone": "osnet_x1_0",
            "embedding_dim": self.expected_dim,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_students": len(self.profiles),
            "students": [self.profiles[sid].to_dict() for sid in self._student_ids_ordered],
        }

        manifest_path = target_dir / "students_registry.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        for sid in self._student_ids_ordered:
            samples_path = emb_dir / f"{sid}_samples.npy"
            proto_path = emb_dir / f"{sid}_prototype.npy"
            np.save(samples_path, self.sample_embeddings[sid].astype(np.float32))
            np.save(proto_path, self.prototype_embeddings[sid].astype(np.float32))

        return manifest_path

    def load(self, registry_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Loads persisted student profiles and embeddings from disk, verifying dimensionality
        and L2 normalization integrity.
        """
        target_dir = Path(registry_dir) if registry_dir is not None else self.registry_dir
        if target_dir is None:
            raise EnrollmentError("No registry_dir specified for loading StudentRegistry.")

        manifest_path = target_dir / "students_registry.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Student registry manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        emb_dir = target_dir / "embeddings"
        loaded_profiles: Dict[str, StudentProfile] = {}
        loaded_rolls: Dict[str, str] = {}
        loaded_samples: Dict[str, np.ndarray] = {}
        loaded_protos: Dict[str, np.ndarray] = {}

        for item in manifest.get("students", []):
            profile = StudentProfile.from_dict(item)
            sid = profile.student_id

            samples_path = emb_dir / f"{sid}_samples.npy"
            proto_path = emb_dir / f"{sid}_prototype.npy"
            if not samples_path.exists() or not proto_path.exists():
                raise EnrollmentError(f"Missing embedding .npy files for student '{sid}' in {emb_dir}")

            raw_samples = np.load(samples_path)
            if raw_samples.ndim == 1:
                raw_samples = raw_samples.reshape(1, -1)
            if raw_samples.shape[1] != self.expected_dim:
                raise EnrollmentError(
                    f"Corrupted sample embedding shape {raw_samples.shape} for '{sid}'."
                )

            norm_samples = np.vstack(
                [normalize_l2(row, expected_dim=self.expected_dim) for row in raw_samples]
            ).astype(np.float32)

            raw_proto = np.load(proto_path)
            norm_proto = normalize_l2(raw_proto, expected_dim=self.expected_dim)

            loaded_profiles[sid] = profile
            loaded_rolls[profile.roll_number] = sid
            loaded_samples[sid] = norm_samples
            loaded_protos[sid] = norm_proto

        self.registry_dir = target_dir
        self.profiles = loaded_profiles
        self.roll_to_student_id = loaded_rolls
        self.sample_embeddings = loaded_samples
        self.prototype_embeddings = loaded_protos
        self._rebuild_indices()

    def query_identity(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computes cosine similarity between a normalized 512-D query embedding and every
        enrolled student in the gallery.
        When hybrid biometric embeddings are present across multiple enrolled students,
        applies Frontal-Anchor Consensus (`f0 >= 0.365`), Within-Class Pose-Subspace
        Orthogonalization (`_depose_sf`), and multi-view rank agreement so side-profile or
        clothing-only similarity can never produce false-positive identity confirmations.
        Returns candidates sorted by descending similarity score.
        """
        if self.is_empty() or self._prototype_matrix is None or self._all_samples_matrix is None:
            return []

        q = normalize_l2(query_embedding, expected_dim=self.expected_dim)

        if getattr(self, "_has_biometric_gallery", False) and self._biometric_gallery:
            from src.reid.reid_model import unpack_hybrid_embedding

            is_bio, sf_q, hd_q, bd_q, q_val = unpack_hybrid_embedding(q)
            q_vec = self._build_centered_vec(sf_q, hd_q, bd_q, is_bio)

            if not is_bio or q_val < 0.40:
                # Faceless crop or degraded face: cap strictly below low_confidence_threshold (0.50)
                # so uniform/body similarity alone can never confirm or vote for a student identity.
                candidates: List[Dict[str, Any]] = []
                for sid in self._student_ids_ordered:
                    g = self._biometric_gallery[sid]
                    p_sim = float(np.dot(q_vec, g["proto"]))
                    m_sim = max(float(np.dot(q_vec, s)) for s in g["samples"])
                    combined_sim = float(np.clip(0.35 * max(0.0, p_sim), 0.0, 0.45))
                    profile = self.profiles[sid]
                    candidates.append({
                        "student_id": sid,
                        "name": profile.name,
                        "roll_number": profile.roll_number,
                        "department": profile.department,
                        "section": profile.section,
                        "similarity": round(combined_sim, 4),
                        "prototype_similarity": round(p_sim, 4),
                        "max_sample_similarity": round(m_sim, 4),
                    })
                candidates.sort(key=lambda x: x["similarity"], reverse=True)
                return candidates

            q_dep = self._depose_sf(sf_q)
            f0_map: Dict[str, float] = {}
            fmax_map: Dict[str, float] = {}
            fmin_map: Dict[str, float] = {}
            dep_map: Dict[str, float] = {}
            psim_map: Dict[str, float] = {}
            msim_map: Dict[str, float] = {}

            for sid in self._student_ids_ordered:
                g = self._biometric_gallery[sid]
                f0_map[sid] = float(np.dot(sf_q, g["frontal_sf"]))
                fmax_map[sid] = max(float(np.dot(sf_q, s)) for s in g["valid_sfs"])
                fmin_map[sid] = min(float(np.dot(sf_q, s)) for s in g["valid_sfs"])
                dep_map[sid] = float(np.dot(q_dep, g["dep_proto_sf"]))
                psim_map[sid] = float(np.dot(q_vec, g["proto"]))
                msim_map[sid] = max(float(np.dot(q_vec, s)) for s in g["samples"])

            top_f0_sid = max(self._student_ids_ordered, key=lambda s: f0_map[s])
            top_fmax_sid = max(self._student_ids_ordered, key=lambda s: fmax_map[s])

            candidates = []
            for sid in self._student_ids_ordered:
                f0 = f0_map[sid]
                fmax = fmax_map[sid]
                fmin = fmin_map[sid]
                dep = dep_map[sid]
                p_sim = psim_map[sid]
                m_sim = msim_map[sid]

                agree = (sid == top_f0_sid and sid == top_fmax_sid) or (
                    sid == top_f0_sid and f0 >= 0.390 and dep >= 0.390
                )
                raw = 0.45 * f0 + 0.35 * dep + 0.20 * fmax
                if f0 >= 0.365 and agree and dep >= 0.365 and fmin >= 0.260 and raw >= 0.380:
                    combined_sim = float(np.clip(0.70 + (raw - 0.380) * 0.90, 0.70, 1.0))
                else:
                    combined_sim = float(min(0.64, 1.55 * max(0.0, f0)))

                profile = self.profiles[sid]
                candidates.append({
                    "student_id": sid,
                    "name": profile.name,
                    "roll_number": profile.roll_number,
                    "department": profile.department,
                    "section": profile.section,
                    "similarity": round(combined_sim, 4),
                    "prototype_similarity": round(p_sim, 4),
                    "max_sample_similarity": round(m_sim, 4),
                })

            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            return candidates

        # Legacy cosine similarity path (synthetic embeddings or single-student galleries)
        proto_sims = np.dot(self._prototype_matrix, q)
        sample_sims = np.dot(self._all_samples_matrix, q)

        max_sample_sim_per_student: Dict[str, float] = {}
        for owner_sid, s_sim in zip(self._sample_owner_ids, sample_sims):
            val = float(s_sim)
            if owner_sid not in max_sample_sim_per_student or val > max_sample_sim_per_student[owner_sid]:
                max_sample_sim_per_student[owner_sid] = val

        candidates = []
        for idx, sid in enumerate(self._student_ids_ordered):
            p_sim = float(proto_sims[idx])
            m_sim = max_sample_sim_per_student.get(sid, p_sim)
            combined_sim = 0.5 * p_sim + 0.5 * m_sim
            profile = self.profiles[sid]
            candidates.append({
                "student_id": sid,
                "name": profile.name,
                "roll_number": profile.roll_number,
                "department": profile.department,
                "section": profile.section,
                "similarity": round(float(combined_sim), 4),
                "prototype_similarity": round(p_sim, 4),
                "max_sample_similarity": round(m_sim, 4),
            })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates
