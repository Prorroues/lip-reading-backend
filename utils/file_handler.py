# -*- coding: utf-8 -*-
import uuid
from pathlib import Path

from fastapi import UploadFile

from utils import settings

CHUNK_SIZE = 4 * 1024 * 1024


class UploadTooLargeError(Exception):
    """上传文件超过大小限制。"""


async def save_file(file: UploadFile, directory: Path) -> str:
    """流式保存上传文件，超过 MAX_UPLOAD_MB 立即中断并删除残留文件。"""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in settings.VIDEO_EXTS:
        suffix = ".mp4"
    dest = directory / f"{uuid.uuid4()}{suffix}"
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLargeError(
                        f"文件超过大小限制 {settings.MAX_UPLOAD_MB}MB"
                    )
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return str(dest)