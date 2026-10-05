# -*- coding: utf-8 -*-
import torch
from .pipelines.pipeline import InferencePipeline


class VSRInference:
    def __init__(self):
        cfg = {
            'config_filename': 'Visual_Speech_Recognition_for_Multiple_Languages/configs/CMLR_V_WER8.0_out.ini',
            'data_dir': None,
            'data_filename': None,
            'data_ext': '.mp4',
            'landmarks_dir': None,
            'landmarks_filename': None,
            'landmarks_ext': '.pkl',
            'labels_filename': None,
            'detector': 'mediapipe',
            'dst_filename': None,
            'gpu_idx': 0,
            'output_subdir': None,
        }
        device = torch.device(
            f"cuda:{cfg['gpu_idx']}" if torch.cuda.is_available() and cfg["gpu_idx"] >= 0 else "cpu"
        )
        self.vsr = InferencePipeline(
            cfg["config_filename"],
            device=device,
            detector=cfg["detector"],
            face_track=True,
        )

    def process_videos(self, video_path, detector="mediapipe"):
        """识别视频中的唇语。

        返回识别文本；没有识别到内容返回空字符串；
        识别失败抛异常（由调用方决定如何处理，不再把错误文本塞进结果里）。
        """
        data = self.vsr.forward(video_path)
        if data is None:
            return ""
        text = str(data).strip()
        return text