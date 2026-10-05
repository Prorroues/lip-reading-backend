# -*- coding: utf-8 -*-
"""统一配置：所有路径、常量、环境变量集中在这里定义。

其他地方不要再写硬编码路径/端口，统一从这里取。
"""
import os
from pathlib import Path

# ---- 项目根目录（以本文件位置推导，与启动目录无关）----
BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
VIDEO_DIR = UPLOAD_DIR / "videos"
ROTATED_VIDEO_DIR = UPLOAD_DIR / "clockwise_videos"
GENERATE_DIR = BASE_DIR / "generates"
GENERATE_VIDEOS_DIR = GENERATE_DIR / "videos"
GENERATE_IMAGES_DIR = GENERATE_DIR / "images"
LOG_DIR = BASE_DIR / "logs"

# ---- 允许的视频扩展名（全项目唯一一份定义）----
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ---- 可调参数（环境变量覆盖，默认值保持与原行为一致）----
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))          # 上传文件大小上限
RETENTION_DAYS = float(os.getenv("RETENTION_DAYS", "3"))        # 上传/产物文件保留天数
INFERENCE_CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "1"))  # 推理并发度（GPU/NPU 建议 1）
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "3600"))   # 已完成任务保留时长
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "600"))  # 处理中任务超时判失败
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")              # 可选：设置后所有敏感接口要求 X-API-Key
DEBUG = os.getenv("DEBUG", "0") == "1"                          # 开启后 500 响应携带 traceback

# Dify 配置（端口统一走这里，不再硬编码 21000）
DIFY_BASE_URL = os.getenv("base_url", "").rstrip("/")
DIFY_PORT = os.getenv("dify_port", "21000")
DIFY_API_KEY = os.getenv("api_key", "")
DIFY_APP_USER = os.getenv("app_user", "abc-123")

# 服务监听
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "24060"))


def ensure_dirs() -> None:
    """启动时创建所有运行时目录。"""
    for d in (VIDEO_DIR, ROTATED_VIDEO_DIR, GENERATE_VIDEOS_DIR, GENERATE_IMAGES_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def validate_env() -> list[str]:
    """启动时检查环境变量，返回警告列表（不直接退出，缺失项只降级对应功能）。"""
    warnings: list[str] = []
    if not DIFY_BASE_URL:
        warnings.append("缺少环境变量 base_url：/llm_interaction 大模型纠错将不可用")
    if not DIFY_API_KEY:
        warnings.append("缺少环境变量 api_key：/llm_interaction 大模型纠错将不可用")
    if not os.getenv("FACE_ESTIMATE_PATH"):
        warnings.append("缺少环境变量 FACE_ESTIMATE_PATH：年龄性别估计将使用 MiVOLO 默认权重路径")
    if not os.getenv("FACE_DETECTOR_PATH"):
        warnings.append("缺少环境变量 FACE_DETECTOR_PATH：人脸检测将使用默认权重路径")
    if not BACKEND_API_KEY:
        warnings.append("未设置 BACKEND_API_KEY：所有接口无鉴权，请勿将服务暴露到不可信网络")
    return warnings


def require_dify_config() -> None:
    """调用 Dify 前校验配置，缺失时给出明确错误（不再 None.rstrip 崩溃）。"""
    missing = []
    if not DIFY_BASE_URL:
        missing.append("base_url")
    if not DIFY_API_KEY:
        missing.append("api_key")
    if missing:
        raise RuntimeError(f"Dify 配置缺失，请在 .env 中配置: {', '.join(missing)}")