#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import cv2
import mediapipe as mp
import numpy as np
import torchvision


class LandmarksDetector:
    """提取唇部对齐用关键点（MediaPipe FaceDetection）。"""

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.35, model_selection=0
        )
        self.full_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.35, model_selection=1
        )

    def __call__(self, filename):
        video_frames = torchvision.io.read_video(filename, pts_unit="sec")[0].numpy()
        frames = [self._prepare_frame(f) for f in video_frames]

        # 眼镜近距离人脸用 short-range；full-range 会先抓住远处海报/旁人。
        landmarks = self.detect(frames, self.short_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(frames, self.full_range_detector)

        return landmarks

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

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            max_id, max_size = 0, 0
            for idx, detected_faces in enumerate(results.detections):
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].y
                            * ih
                        ),
                    ],
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks
