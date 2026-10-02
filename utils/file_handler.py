import uuid
from pathlib import Path

from fastapi import UploadFile

CHUNK_SIZE = 4 * 1024 * 1024


async def save_file(file: UploadFile, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"
    dest = directory / f"{uuid.uuid4()}{suffix}"
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
    return str(dest)
