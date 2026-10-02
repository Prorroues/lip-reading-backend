
from pathlib import Path


def file_name_extract(file_path):
    """从上传文件路径推导生成产物路径（跨平台）。"""
    file_name = Path(file_path).stem
    return (
        str(Path("generates") / "videos" / f"{file_name}.mp4"),
        str(Path("generates") / "images" / f"{file_name}.png"),
        str(Path("uploads") / "clockwise_videos" / f"{file_name}.mp4"),
    )
