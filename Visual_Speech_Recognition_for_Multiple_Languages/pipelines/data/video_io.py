# -*- coding: utf-8 -*-
"""读视频帧：torchvision 失败时用 OpenCV（眼镜 HEVC 常把 PyAV 打崩）。"""

import logging

import cv2
import numpy as np
import torchvision

logger = logging.getLogger("vsr.video_io")


def _probe_fps(path, info=None, default=25.0):
    fps = None
    if isinstance(info, dict):
        fps = info.get("video_fps")
    if not fps or fps <= 1:
        cap = cv2.VideoCapture(path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
    if not fps or fps <= 1:
        return default
    return float(fps)


def _load_with_cv2(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"视频没有可读帧: {path}")
    return np.stack(frames), float(fps)


def load_video_rgb(path):
    """返回 (RGB uint8 [T,H,W,3], fps)。"""
    try:
        frames, _, info = torchvision.io.read_video(path, pts_unit="sec")
        if frames is not None and frames.numel() > 0:
            return frames.numpy(), _probe_fps(path, info)
        logger.warning("torchvision 读到空帧，改用 OpenCV: %s", path)
    except Exception as e:
        logger.warning("torchvision 读视频失败，改用 OpenCV: %s (%s)", path, e)
    return _load_with_cv2(path)
