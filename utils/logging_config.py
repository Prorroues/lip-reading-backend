# -*- coding: utf-8 -*-
"""日志配置：控制台 + 滚动文件双输出，替代散落的 print。"""
import logging
from logging.handlers import RotatingFileHandler

from utils import settings

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.LOG_DIR / "backend.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True