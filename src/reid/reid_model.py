from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("trace.reid.model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def _unit_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).flatten()
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else np.zeros_like(v, dtype=np.float32)


def pool_512_to_128(vec_512: np.ndarray) -> np.ndarray:
    """Pools a 512-D OSNet embedding into a 128-D L2-normalized descriptor."""
    v = np.asarray(vec_512, dtype=np.float32).reshape(128, 4).mean(axis=1)
    return _unit_vec(v)


def compute_face_pose_quality(face_row: np.ndarray) -> float:
    """
    Computes facial landmark symmetry and frontalness quality in [0.05, 1.0]
    from YuNet 15-float detection output:
    [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score].
    """
    w = max(1.0, float(face_row[2]))
    h = max(1.0, float(face_row[3]))
    re_x, re_y = float(face_row[4]), float(face_row[5])
    le_x, le_y = float(face_row[6]), float(face_row[7])
    nt_x, nt_y = float(face_row[8]), float(face_row[9])
    rcm_x, rcm_y = float(face_row[10]), float(face_row[11])
    lcm_x, lcm_y = float(face_row[12]), float(face_row[13])
    score = float(face_row[14])

    eye_dist_ratio = abs(le_x - re_x) / w
    frontal_eye = float(np.clip((eye_dist_ratio - 0.12) / 0.28, 0.08, 1.0))

    eye_mid_x = 0.5 * (re_x + le_x)
    eye_span = max(1.0, abs(le_x - re_x))
    nose_offset = abs(nt_x - eye_mid_x) / eye_span
    symmetry = float(np.clip(1.0 - 0.9 * nose_offset, 0.15, 1.0))

    eye_mid_y = 0.5 * (re_y + le_y)
    mouth_mid_y = 0.5 * (rcm_y + lcm_y)
    vert_valid = 1.0 if (eye_mid_y < nt_y < mouth_mid_y and (mouth_mid_y - eye_mid_y) > 0.20 * h) else 0.15

    return float(score * frontal_eye * symmetry * vert_valid)


def pack_hybrid_embedding(
    sface_128: np.ndarray,
    head_128: np.ndarray,
    body_128: np.ndarray,
    pose_quality: float = 0.85,
    has_face: bool = True,
    w_face: float = 0.75,
    w_head: float = 0.15,
    w_body: float = 0.10,
) -> np.ndarray:
    """
    Packs a 128-D SFace biometric vector, pose quality q in [0.05, 1.0],
    128-D OSNet head vector, and 128-D OSNet upper-body vector into a single
    unit L2-normalized 512-D float32 vector.
    """
    hd = _unit_vec(head_128)
    bd = _unit_vec(body_128)
    if has_face and float(np.linalg.norm(sface_128)) > 1e-6:
        sf = _unit_vec(sface_128)
        q = float(np.clip(pose_quality, 0.05, 1.0))
        theta = (np.pi / 4.0) * q
        wf = np.sqrt(w_face)
        wh = np.sqrt(w_head)
        wb = np.sqrt(w_body)
        vec = np.concatenate([
            wf * np.sin(theta) * sf,
            wf * np.cos(theta) * sf,
            wh * hd,
            wb * bd,
        ]).astype(np.float32)
    else:
        vec = np.concatenate([
            np.zeros(256, dtype=np.float32),
            np.sqrt(0.55) * hd,
            np.sqrt(0.45) * bd,
        ]).astype(np.float32)
    return _unit_vec(vec)


def unpack_hybrid_embedding(
    vec_512: np.ndarray,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Losslessly unpacks a 512-D hybrid embedding into:
    (is_biometric_face, sface_128, head_128, body_128, pose_quality).
    Returns is_biometric_face=False for synthetic test vectors or faceless crops.
    """
    v = np.asarray(vec_512, dtype=np.float32).flatten()
    if v.shape[0] != 512:
        z = np.zeros(128, dtype=np.float32)
        return False, z, z, z, 0.05

    b0 = v[0:128]
    b1 = v[128:256]
    hd = _unit_vec(v[256:384])
    bd = _unit_vec(v[384:512])

    n0 = float(np.linalg.norm(b0))
    n1 = float(np.linalg.norm(b1))
    if n0 > 0.02 and n1 > 0.02:
        u0 = b0 / n0
        u1 = b1 / n1
        collinear_sim = float(np.dot(u0, u1))
        if collinear_sim > 0.985:
            sf = _unit_vec(b0 + b1)
            q = float(np.clip((4.0 / np.pi) * np.arctan2(n0, n1), 0.05, 1.0))
            return True, sf, hd, bd, q

    return False, np.zeros(128, dtype=np.float32), hd, bd, 0.05


class ReIDModel:
    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(f"Using device: {self.device}")

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True,
            use_gpu=(self.device.type == "cuda"),
        )

        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.face_detector = None
        self.face_recognizer = None
        self._init_biometric_models()

    def _init_biometric_models(self) -> None:
        """Initializes OpenCV YuNet face detector and SFace recognizer if available."""
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            return
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            yunet_path = MODELS_DIR / YUNET_FILENAME
            sface_path = MODELS_DIR / SFACE_FILENAME
            if not yunet_path.exists():
                urllib.request.urlretrieve(YUNET_URL, str(yunet_path))
            if not sface_path.exists():
                urllib.request.urlretrieve(SFACE_URL, str(sface_path))
            self.face_detector = cv2.FaceDetectorYN.create(
                str(yunet_path), "", (320, 320), 0.40, 0.3, 5000
            )
            self.face_recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        except Exception as exc:
            logger.warning("Biometric face models unavailable, falling back to OSNet only: %s", exc)
            self.face_detector = None
            self.face_recognizer = None

    def _to_bgr_array(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                return image_input.copy()
            pil_img = Image.fromarray(image_input).convert("RGB")
            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _extract_osnet_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        tensors = []
        for c in crops_bgr:
            if c is None or c.size == 0:
                c = np.zeros((64, 32, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = F.normalize(self.model(batch), p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def _detect_primary_face(
        self, img_bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        if self.face_detector is None or self.face_recognizer is None:
            return None
        h, w = img_bgr.shape[:2]
        if h < 24 or w < 24:
            return None

        search_h = max(24, int(h * 0.75))
        search_img = img_bgr[0:search_h, :]
        sh, sw = search_img.shape[:2]

        scale = 1.0
        max_dim = max(sh, sw)
        if max_dim > 800:
            scale = 800.0 / max_dim
            work = cv2.resize(search_img, (int(sw * scale), int(sh * scale)))
        elif max_dim < 180:
            scale = 240.0 / max_dim
            work = cv2.resize(search_img, (int(sw * scale), int(sh * scale)))
        else:
            work = search_img

        wh, ww = work.shape[:2]
        self.face_detector.setInputSize((ww, wh))
        _, faces = self.face_detector.detect(work)
        if faces is None or len(faces) == 0:
            return None

        best_f = None
        best_m = -1.0
        for f in faces:
            score = float(f[-1])
            if score < 0.42:
                continue
            fx, fy, fw, fh = f[:4]
            cx = (fx + fw / 2.0) / ww
            cy = (fy + fh / 2.0) / wh
            if cy > 0.80:
                continue
            pq = compute_face_pose_quality(f)
            area_ratio = max(1e-4, (fw * fh) / float(ww * wh))
            center_w = max(0.25, 1.0 - abs(cx - 0.5) * 1.1)
            m = (area_ratio ** 0.45) * center_w * (0.45 + 0.55 * pq)
            if m > best_m:
                best_m = m
                best_f = f.copy()

        if best_f is None:
            return None

        orig_fw = float(best_f[2]) / scale
        orig_fh = float(best_f[3]) / scale
        if orig_fw < 18.0 or orig_fh < 18.0:
            return None

        aligned = self.face_recognizer.alignCrop(work, best_f)
        sface_feat = self.face_recognizer.feature(aligned).flatten().astype(np.float32)
        sface_feat = _unit_vec(sface_feat)
        pq = compute_face_pose_quality(best_f)
        q = float(0.35 + 0.65 * pq)
        box = tuple((best_f[:4] / scale).astype(int).tolist())
        return sface_feat, q, box  # type: ignore[return-value]

    def extract_biometric_fast(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Fast biometric extraction path for video track crops when matching against a
        biometric student gallery. Runs YuNet+SFace first and skips the expensive 2-crop
        OSNet GPU forward pass when no valid face (fw >= 18px, pose_quality >= 0.40) is present.
        """
        if self.face_detector is None or self.face_recognizer is None:
            return self.extract_embedding(image_input)

        img_bgr = self._to_bgr_array(image_input)
        h, w = img_bgr.shape[:2]
        face_res = self._detect_primary_face(img_bgr)
        if face_res is None:
            placeholder = np.zeros(128, dtype=np.float32)
            placeholder[0] = 1.0
            return pack_hybrid_embedding(
                sface_128=np.zeros(128, dtype=np.float32),
                head_128=placeholder,
                body_128=placeholder,
                pose_quality=0.05,
                has_face=False,
            )

        sface_feat, q, (fx, fy, fw, fh) = face_res
        if q < 0.40 or fw < 18 or fh < 18:
            placeholder = np.zeros(128, dtype=np.float32)
            placeholder[0] = 1.0
            return pack_hybrid_embedding(
                sface_128=np.zeros(128, dtype=np.float32),
                head_128=placeholder,
                body_128=placeholder,
                pose_quality=0.05,
                has_face=False,
            )

        hx1, hy1 = max(0, int(fx - 0.42 * fw)), max(0, int(fy - 0.42 * fh))
        hx2, hy2 = min(w, int(fx + 1.42 * fw)), min(h, int(fy + 1.60 * fh))
        head_crop = img_bgr[hy1:hy2, hx1:hx2]
        if head_crop.size == 0:
            head_crop = img_bgr[0:max(1, int(h * 0.45)), :]

        ux1, uy1 = max(0, int(fx - 1.15 * fw)), max(0, int(fy - 0.40 * fh))
        ux2, uy2 = min(w, int(fx + 2.15 * fw)), min(h, int(fy + 3.20 * fh))
        upper_crop = img_bgr[uy1:uy2, ux1:ux2]
        if upper_crop.size == 0:
            upper_crop = img_bgr[0:max(1, int(h * 0.65)), :]

        osnet_feats = self._extract_osnet_batch([head_crop, upper_crop])
        head_128 = pool_512_to_128(osnet_feats[0])
        body_128 = pool_512_to_128(osnet_feats[1])
        return pack_hybrid_embedding(
            sface_128=sface_feat,
            head_128=head_128,
            body_128=body_128,
            pose_quality=q,
            has_face=True,
        )

    def extract_embedding(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extracts a 512-D L2-normalized hybrid biometric + multi-region OSNet embedding from:
        - File path (str or Path)
        - PIL.Image.Image instance
        - np.ndarray (in-memory crop, BGR or RGB)

        When a face is visible in the image/crop, isolates the foreground target head and
        upper-body regions around the primary face and fuses the 128-D SFace facial biometric
        descriptor with 128-D head OSNet and 128-D upper-body OSNet descriptors into a unit
        512-D vector. Falls back to regional/holistic OSNet when no face is visible.
        """
        img_bgr = self._to_bgr_array(image_input)
        h, w = img_bgr.shape[:2]

        face_res = self._detect_primary_face(img_bgr)
        if face_res is not None:
            sface_feat, q, (fx, fy, fw, fh) = face_res
            hx1, hy1 = max(0, int(fx - 0.42 * fw)), max(0, int(fy - 0.42 * fh))
            hx2, hy2 = min(w, int(fx + 1.42 * fw)), min(h, int(fy + 1.60 * fh))
            head_crop = img_bgr[hy1:hy2, hx1:hx2]
            if head_crop.size == 0:
                head_crop = img_bgr[0:max(1, int(h * 0.45)), :]

            ux1, uy1 = max(0, int(fx - 1.15 * fw)), max(0, int(fy - 0.40 * fh))
            ux2, uy2 = min(w, int(fx + 2.15 * fw)), min(h, int(fy + 3.20 * fh))
            upper_crop = img_bgr[uy1:uy2, ux1:ux2]
            if upper_crop.size == 0:
                upper_crop = img_bgr[0:max(1, int(h * 0.65)), :]

            osnet_feats = self._extract_osnet_batch([head_crop, upper_crop])
            head_128 = pool_512_to_128(osnet_feats[0])
            body_128 = pool_512_to_128(osnet_feats[1])
            return pack_hybrid_embedding(
                sface_128=sface_feat,
                head_128=head_128,
                body_128=body_128,
                pose_quality=q,
                has_face=True,
            )

        # Fallback for faceless crops or synthetic unit-test images
        head_crop = img_bgr[0:max(1, int(h * 0.40)), :]
        upper_crop = img_bgr[0:max(1, int(h * 0.65)), :]
        osnet_feats = self._extract_osnet_batch([head_crop, upper_crop])
        head_128 = pool_512_to_128(osnet_feats[0])
        body_128 = pool_512_to_128(osnet_feats[1])
        return pack_hybrid_embedding(
            sface_128=np.zeros(128, dtype=np.float32),
            head_128=head_128,
            body_128=body_128,
            pose_quality=0.05,
            has_face=False,
        )


def get_first_reference_image():
    image_dir = Path("data/raw/reference_images")
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []

    for ext in extensions:
        image_files.extend(image_dir.glob(ext))

    if not image_files:
        raise FileNotFoundError(
            "No reference images found in data/raw/reference_images/"
        )

    image_files.sort()
    return str(image_files[0])


if __name__ == "__main__":
    model = ReIDModel()
    image_path = get_first_reference_image()
    print(f"Testing with: {image_path}")

    embedding = model.extract_embedding(image_path)
    print("-------------------------------------")
    print("Embedding generated successfully.")
    print(f"Embedding Shape : {embedding.shape}")
    print(f"Embedding dtype : {embedding.dtype}")
    print("-------------------------------------")
