# -*- coding: utf-8 -*-
"""上传/产物目录定期清理：防止磁盘只进不出被写满。"""
import asyncio
import logging
import time
from pathlib import Path

from utils import settings

logger = logging.getLogger("backend.cleanup")

_CLEAN_INTERVAL = 3600  # 每小时扫一次


def _cleanup_dir(directory: Path, max_age_seconds: float) -> int:
    if not directory.is_dir():
        return 0
    now = time.time()
    removed = 0
    for f in directory.iterdir():
        try:
            if f.is_file() and now - f.stat().st_mtime > max_age_seconds:
                f.unlink()
                removed += 1
        except OSError as e:
            logger.warning("清理文件失败 %s: %s", f, e)
    return removed


async def cleanup_loop(stop_event: asyncio.Event) -> None:
    """后台协程：周期性清理过期文件，stop_event 置位后退出。"""
    max_age = settings.RETENTION_DAYS * 86400
    while not stop_event.is_set():
        total = 0
        for d in (
            settings.VIDEO_DIR,
            settings.ROTATED_VIDEO_DIR,
            settings.GENERATE_VIDEOS_DIR,
            settings.GENERATE_IMAGES_DIR,
        ):
            total += _cleanup_dir(d, max_age)
        if total:
            logger.info("定期清理完成，删除过期文件 %d 个", total)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_CLEAN_INTERVAL)
        except asyncio.TimeoutError:
            pass