#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from .transforms import AudioTransform, VideoTransform
from .video_io import load_video_rgb

logger = logging.getLogger("vsr.data")


def save_mouth_roi(frames, dump_path, fps):
    """把 96x96 灰度嘴块存成可回放 mp4，方便对照裁切是否居中。"""
    path = Path(dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames is None or len(frames) == 0:
        return
    first = frames[0]
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (int(width), int(height)),
    )
    try:
        for frame in frames:
            if frame.ndim == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                bgr = cv2.cvtColor(np.squeeze(frame, axis=-1), cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[-1] == 3:
                bgr = frame
            else:
                continue
            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)
            writer.write(bgr)
    finally:
        writer.release()


class AVSRDataLoader:
    def __init__(self, modality, speed_rate=1, transform=True, detector="retinaface", convert_gray=True, model_fps=25.0):
        self.modality = modality
        self.transform = transform
        self.fallback_speed_rate = speed_rate
        self.model_fps = float(model_fps) if model_fps else 25.0
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform()
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from ..detectors.mediapipe.video_process import VideoProcess
                self.video_process = VideoProcess(convert_gray=convert_gray)
            if detector == "retinaface":
                from ..detectors.retinaface.video_process import VideoProcess
                self.video_process = VideoProcess(convert_gray=convert_gray)
            self.video_transform = VideoTransform(speed_rate=speed_rate)

    def _video_transform_for(self, src_fps):
        speed_rate = src_fps / self.model_fps if src_fps and self.model_fps else self.fallback_speed_rate
        if abs(speed_rate - self.fallback_speed_rate) < 1e-3 and hasattr(self, "video_transform"):
            return self.video_transform, speed_rate
        return VideoTransform(speed_rate=speed_rate), speed_rate

    def load_data(self, data_filename, landmarks=None, transform=True, mouth_dump_path=None):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return self.audio_transform(audio) if self.transform else audio
        if self.modality == "video":
            video, src_fps = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            if video is None:
                raise Exception("未能裁出嘴部，请把人脸放在画面中间再录")
            if mouth_dump_path:
                try:
                    save_mouth_roi(video, mouth_dump_path, src_fps)
                except Exception:
                    logger.exception("保存嘴部裁切失败: %s", mouth_dump_path)
            video_tf, speed_rate = self._video_transform_for(src_fps)
            logger.info(
                "嘴部裁切 fps=%.2f model_fps=%.2f speed_rate=%.3f frames=%d dump=%s",
                src_fps, self.model_fps, speed_rate, len(video), mouth_dump_path or "-",
            )
            video = torch.tensor(video)
            return video_tf(video) if self.transform else video
        if self.modality == "audiovisual":
            rate_ratio = 640
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            video, src_fps = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            if video is None:
                raise Exception("未能裁出嘴部，请把人脸放在画面中间再录")
            video_tf, _ = self._video_transform_for(src_fps)
            video = torch.tensor(video)
            min_t = min(len(video), audio.size(1) // rate_ratio)
            audio = audio[:, :min_t*rate_ratio]
            video = video[:min_t]
            if self.transform:
                audio = self.audio_transform(audio)
                video = video_tf(video)
            return video, audio

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return load_video_rgb(data_filename)


    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
