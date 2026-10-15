#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Face Mesh → 68-point landmarks for official-style mouth crop.

4-point FaceDetection backup:
  pipelines/detectors/mediapipe/_backup_4point_2026-10-15/
"""

import cv2
import mediapipe as mp
import numpy as np

from ...data.video_io import load_video_rgb

# MediaPipe Face Mesh 468 → dlib 68, same order as official mean-face / RetinaFace crop.
_MP468_TO_DLIB68 = [
    127, 234, 93, 132, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    78, 82, 13, 312, 308, 317, 14, 87,
]


class LandmarksDetector:
    """提取 68 点关键点（MediaPipe Face Mesh），供按嘴唇轮廓对齐裁切。"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, filename):
        video_frames, _ = load_video_rgb(filename)
        frames = [self._prepare_frame(f) for f in video_frames]
        return [self._detect_one(frame) for frame in frames]

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return np.ascontiguousarray(frame)

    def _detect_one(self, frame: np.ndarray):
        results = self.mesh.process(frame)
        if not results.multi_face_landmarks:
            return None
        ih, iw = frame.shape[:2]
        best = None
        best_size = -1
        for face in results.multi_face_landmarks:
            pts68 = self._to_dlib68(face, iw, ih)
            xs, ys = pts68[:, 0], pts68[:, 1]
            size = (xs.max() - xs.min()) + (ys.max() - ys.min())
            if size > best_size:
                best_size = size
                best = pts68
        return best

    @staticmethod
    def _to_dlib68(face_landmarks, iw, ih) -> np.ndarray:
        pts = []
        for idx in _MP468_TO_DLIB68:
            lm = face_landmarks.landmark[idx]
            pts.append([lm.x * iw, lm.y * ih])
        return np.asarray(pts, dtype=np.float32)
