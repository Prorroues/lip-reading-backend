# -*- coding: utf-8 -*-
"""视频旋转：优先调 ffmpeg 转置（快），失败时回退 OpenCV 逐帧旋转。

历史说明：旧版两个函数重复 95% 且逆时针分支忘记 release，已合并修复。
"""
import logging
import shutil
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger("backend.rotate")

_DIRECTIONS = {"cw", "ccw"}
# ffmpeg transpose: 1=顺时针90°, 2=逆时针90°
_FFMPEG_TRANSPOSE = {"cw": "1", "ccw": "2"}
_CV2_ROTATE = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}


def _rotate_with_ffmpeg(input_path: str, output_path: str, direction: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", f"transpose={_FFMPEG_TRANSPOSE[direction]}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("ffmpeg 旋转失败，回退 OpenCV: %s", e)
        Path(output_path).unlink(missing_ok=True)
        return False


def _rotate_with_opencv(input_path: str, output_path: str, direction: str) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_path}")
    out = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 旋转后宽高互换
        height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"无法初始化视频写出器: {output_path}")
        rotate_code = _CV2_ROTATE[direction]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(cv2.rotate(frame, rotate_code))
    finally:
        cap.release()
        if out is not None:
            out.release()


def rotate_video(input_path: str, output_path: str, direction: str) -> None:
    """把视频旋转 90°。direction: "cw"=顺时针, "ccw"=逆时针。"""
    if direction not in _DIRECTIONS:
        raise ValueError(f"非法旋转方向: {direction!r}，应为 {_DIRECTIONS}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if _rotate_with_ffmpeg(input_path, output_path, direction):
        return
    _rotate_with_opencv(input_path, output_path, direction)


# ---- 兼容旧接口名（新代码请直接用 rotate_video）----
def rotate_video_clockwise_sync(input_path: str, output_path: str) -> None:
    rotate_video(input_path, output_path, "cw")


def rotate_video_counterclockwise_sync(input_path: str, output_path: str) -> None:
    rotate_video(input_path, output_path, "ccw")