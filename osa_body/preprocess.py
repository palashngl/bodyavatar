from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import ConvexHull
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "pose_landmarker_lite.task"


def ensure_pose_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class BodyPoseTracker:
    """Extract per-frame body pose landmarks and normalized full-body crops."""

    NUM_LMK = 33

    def __init__(self, image_size: int = 384) -> None:
        self.image_size = image_size

    def _create_landmarker(self) -> vision.PoseLandmarker:
        model_path = ensure_pose_model()
        opts = vision.PoseLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(opts)

    def _landmarks_from_result(
        self, result, crop_w: int, crop_h: int, offset_x: int, offset_y: int
    ) -> np.ndarray | None:
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        if len(lms) < self.NUM_LMK:
            return None
        pts = np.zeros((self.NUM_LMK, 3), dtype=np.float32)
        for i in range(self.NUM_LMK):
            lm = lms[i]
            if lm.visibility < 0.35:
                pts[i, 0] = np.nan
                pts[i, 1] = np.nan
                pts[i, 2] = 0.0
                continue
            pts[i, 0] = lm.x * crop_w + offset_x
            pts[i, 1] = lm.y * crop_h + offset_y
            pts[i, 2] = lm.z * crop_w
        if np.isnan(pts[:, :2]).all():
            return None
        return pts

    def _bbox_from_landmarks(self, pts: np.ndarray, w: int, h: int, pad: float = 0.18) -> tuple[int, int, int, int]:
        valid = pts[np.isfinite(pts[:, 0])]
        if len(valid) == 0:
            return 0, 0, w, h
        x0, y0 = valid[:, 0].min(), valid[:, 1].min()
        x1, y1 = valid[:, 0].max(), valid[:, 1].max()
        bw, bh = x1 - x0, y1 - y0
        x0 = int(max(0, x0 - pad * bw))
        y0 = int(max(0, y0 - pad * bh))
        x1 = int(min(w, x1 + pad * bw))
        y1 = int(min(h, y1 + pad * bh))
        return x0, y0, max(x1 - x0, 32), max(y1 - y0, 32)

    def _fill_missing(self, pts: np.ndarray) -> np.ndarray:
        out = pts.copy()
        mean_x = np.nanmean(out[:, 0])
        mean_y = np.nanmean(out[:, 1])
        for i in range(len(out)):
            if not np.isfinite(out[i, 0]):
                out[i, 0] = mean_x
            if not np.isfinite(out[i, 1]):
                out[i, 1] = mean_y
            if not np.isfinite(out[i, 2]):
                out[i, 2] = 0.0
        return out

    def _landmark_mask(self, lmk: np.ndarray, size: int) -> np.ndarray:
        """Soft foreground mask from pose convex hull."""
        pts = lmk[np.isfinite(lmk[:, 0])][:, :2]
        if len(pts) < 3:
            mask = np.zeros((size, size), dtype=np.float32)
            mask[: size // 2, :] = 1.0
            return mask
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices].astype(np.int32)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_pts, 255)
        mask = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        return np.clip(mask, 0.05, 1.0)

    def _normalize_frame(
        self, frame_rgb: np.ndarray, pts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame_rgb.shape[:2]
        x, y, bw, bh = self._bbox_from_landmarks(pts, w, h)
        crop = frame_rgb[y : y + bh, x : x + bw]
        scale = self.image_size / max(bw, bh)
        nh, nw = int(bh * scale), int(bw * scale)
        crop = cv2.resize(crop, (nw, nh))
        pad_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        pad_img[:nh, :nw] = crop
        lmk = pts.copy()
        lmk[:, 0] = (lmk[:, 0] - x) * scale
        lmk[:, 1] = (lmk[:, 1] - y) * scale
        lmk[:, 0] = np.clip(lmk[:, 0], 0, self.image_size - 1)
        lmk[:, 1] = np.clip(lmk[:, 1], 0, self.image_size - 1)
        return pad_img, self._fill_missing(lmk)

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        max_frames: int = 240,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        frames: list[np.ndarray] = []
        landmarks: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        idx = 0
        frame_ts = 0
        landmarker = self._create_landmarker()
        while cap.isOpened() and idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(
                mp_image, int(frame_ts * 1000 / max(fps, 1))
            )
            frame_ts += 1
            pts = self._landmarks_from_result(result, w, h, 0, 0)
            if pts is None:
                continue
            norm_rgb, norm_lmk = self._normalize_frame(rgb, pts)
            frames.append(norm_rgb)
            landmarks.append(norm_lmk)
            masks.append(self._landmark_mask(norm_lmk, self.image_size))
            idx += 1
        cap.release()
        landmarker.close()

        if len(frames) < 16:
            raise RuntimeError(f"Too few body frames in {video_path}: {len(frames)}")

        frames_np = np.stack(frames).astype(np.float32) / 255.0
        landmarks_np = np.stack(landmarks).astype(np.float32)
        masks_np = np.stack(masks).astype(np.float32)
        neutral = np.nanmean(landmarks_np, axis=0)
        neutral = self._fill_missing(neutral)

        np.savez_compressed(
            output_dir / "processed.npz",
            frames=frames_np,
            landmarks=landmarks_np,
            masks=masks_np,
            neutral_landmarks=neutral,
            image_size=np.array([self.image_size], dtype=np.int32),
        )
        meta = {
            "num_frames": len(frames),
            "fps": fps,
            "image_size": self.image_size,
            "source_video": str(video_path),
            "num_landmarks": self.NUM_LMK,
            "task": "full_body_avatar",
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


class LiveBodyPoseTracker(BodyPoseTracker):
    """Reuse one landmarker for webcam / streaming (IMAGE mode)."""

    def __init__(self, image_size: int = 384) -> None:
        super().__init__(image_size=image_size)
        model_path = ensure_pose_model()
        opts = vision.PoseLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        pts = self._landmarks_from_result(result, w, h, 0, 0)
        if pts is None:
            return None
        norm_rgb, norm_lmk = self._normalize_frame(rgb, pts)
        mask = self._landmark_mask(norm_lmk, self.image_size)
        frame_f = norm_rgb.astype(np.float32) / 255.0
        return frame_f, norm_lmk.astype(np.float32), mask

    def close(self) -> None:
        self._landmarker.close()
