# -*- coding: utf-8 -*-
from pathlib import Path

from utils import settings


def file_name_extract(file_path):
    """从上传文件路径推导生成产物路径（绝对路径，跨平台，与启动目录无关）。"""
    stem = Path(file_path).stem
    return (
        str(settings.GENERATE_VIDEOS_DIR / f"{stem}.mp4"),
        str(settings.GENERATE_IMAGES_DIR / f"{stem}.png"),
        str(settings.ROTATED_VIDEO_DIR / f"{stem}.mp4"),
    )