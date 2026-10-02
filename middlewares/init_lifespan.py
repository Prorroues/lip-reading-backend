from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# fastapi_infer 内部使用 from mivolo.predictor，需把 MiVOLO 目录加入 path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MIVOLO_ROOT = os.path.join(_PROJECT_ROOT, "MiVOLO")
if _MIVOLO_ROOT not in sys.path:
    sys.path.insert(0, _MIVOLO_ROOT)

from MiVOLO.fastapi_infer import MivoloInference
from Visual_Speech_Recognition_for_Multiple_Languages.fastapi_vsr import VSRInference
from utils.lip_videos import LipVideo


@asynccontextmanager
async def tai_middleware(app: FastAPI):

    app.state.vsr = VSRInference()
    app.state.executor = ThreadPoolExecutor()
    ckpt_path = os.getenv("FACE_ESTIMATE_PATH")
    det_path = os.getenv("FACE_DETECTOR_PATH")

    app.state.age_estimate = MivoloInference(checkpoint_path=ckpt_path,
                            detector_weights_path=det_path)
    app.state.lip_video = LipVideo()

    try:
        yield  # 应用运行期间
    finally:
        app.state.executor.shutdown(wait=False)
