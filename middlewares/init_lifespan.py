# -*- coding: utf-8 -*-
import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI

# fastapi_infer 内部使用 from mivolo.predictor，需把 MiVOLO 目录加入 path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MIVOLO_ROOT = os.path.join(_PROJECT_ROOT, "MiVOLO")
if _MIVOLO_ROOT not in sys.path:
    sys.path.insert(0, _MIVOLO_ROOT)

from MiVOLO.fastapi_infer import MivoloInference
from Visual_Speech_Recognition_for_Multiple_Languages.fastapi_vsr import VSRInference
from utils import settings
from utils.file_cleanup import cleanup_loop
from utils.lip_videos import LipVideo
from utils.logging_config import setup_logging


@asynccontextmanager
async def tai_middleware(app: FastAPI):
    setup_logging()
    logger = logging.getLogger("backend.lifespan")

    settings.ensure_dirs()
    for w in settings.validate_env():
        logger.warning("配置检查: %s", w)

    app.state.vsr = VSRInference()
    app.state.executor = ThreadPoolExecutor()
    # 推理并发信号量：GPU/NPU 场景建议保持 1，避免显存互抢
    app.state.infer_sem = asyncio.Semaphore(settings.INFERENCE_CONCURRENCY)
    # 全局复用的 HTTP 会话（连接池），供 Dify 等外部调用使用
    app.state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    )

    ckpt_path = os.getenv("FACE_ESTIMATE_PATH")
    det_path = os.getenv("FACE_DETECTOR_PATH")
    app.state.age_estimate = MivoloInference(
        checkpoint_path=ckpt_path, detector_weights_path=det_path
    )
    app.state.lip_video = LipVideo()

    # 过期文件定期清理
    stop_cleanup = asyncio.Event()
    cleanup_task = asyncio.create_task(cleanup_loop(stop_cleanup))
    logger.info("后端启动完成（推理并发度=%d，文件保留 %.1f 天）",
                settings.INFERENCE_CONCURRENCY, settings.RETENTION_DAYS)

    try:
        yield  # 应用运行期间
    finally:
        stop_cleanup.set()
        try:
            await asyncio.wait_for(cleanup_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        await app.state.http_session.close()
        app.state.executor.shutdown(wait=False)