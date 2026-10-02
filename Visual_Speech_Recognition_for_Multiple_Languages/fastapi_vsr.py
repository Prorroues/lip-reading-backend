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
        try:
            data = self.vsr.forward(video_path)
            if data is None or (isinstance(data, str) and not str(data).strip()):
                return "唇语识别结果为空"
            return data
        except Exception as e:
            return f"唇语识别失败: {e}"
