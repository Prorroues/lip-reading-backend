# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import mimetypes
import os
import sys
import traceback
from pathlib import Path

import uvicorn
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form,
                     HTTPException, Query, Request, Response, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from middlewares.init_lifespan import tai_middleware
from pydantic_models.request_models import LLMInteractionModel
from utils import settings, video_tasks
from utils.dify_upload import run_workflow, upload_image_file
from utils.edit_distance import pinyin_similarity_score
from utils.file_handler import UploadTooLargeError, save_file
from utils.file_name_extract import file_name_extract
from utils.latest_recognition import attach_plain_text, get_latest
from utils.rotate_videos import rotate_video
from utils.security import safe_resolve, verify_api_key

logger = logging.getLogger("backend.app")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Visual_Speech_Recognition_for_Multiple_Languages')))
app = FastAPI(lifespan=tai_middleware)

# TODO(安全): 生产环境应把 allow_origins 收紧为实际前端来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(status_code: int, message: str, exc: Exception | None = None) -> JSONResponse:
    """统一错误响应：客户端只看摘要，详细堆栈只进日志（DEBUG 模式才回传）。"""
    if exc is not None:
        logger.error("%s: %s", message, traceback.format_exc())
    content = {"error": message}
    if settings.DEBUG and exc is not None:
        content["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=status_code, content=content)


def _check_upload_basic(request: Request, file: UploadFile) -> JSONResponse | None:
    """上传前置校验：类型、扩展名、Content-Length。不合法返回错误响应，合法返回 None。"""
    if not file.content_type or not file.content_type.startswith("video/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"文件不是视频类型，收到类型: {file.content_type}", "filename": file.filename},
        )
    if file.filename:
        if Path(file.filename).suffix.lower() not in settings.VIDEO_EXTS:
            return JSONResponse(
                status_code=400,
                content={"error": f"不支持的文件格式: {file.filename}，只支持: {', '.join(sorted(settings.VIDEO_EXTS))}"},
            )
    content_length = request.headers.get("content-length")
    try:
        too_big = content_length is not None and int(content_length) > settings.MAX_UPLOAD_MB * 1024 * 1024
    except ValueError:
        too_big = False
    if too_big:
        return JSONResponse(
            status_code=413,
            content={"error": f"文件超过大小限制 {settings.MAX_UPLOAD_MB}MB"},
        )
    return None


def _serve_media(directory: Path, filename: str) -> FileResponse:
    """提供产物文件下载，路径被严格限制在产物目录内。"""
    file_path = safe_resolve(directory, filename)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type, _ = mimetypes.guess_type(file_path)
    if not media_type:
        media_type = "application/octet-stream"
    return FileResponse(path=file_path, media_type=media_type, filename=filename)


@app.post("/upload/video", tags=["上传要检测的视频"], deprecated=True,
          dependencies=[Depends(verify_api_key)])
async def upload_video(request: Request, file: UploadFile = File(...), direction: str = Form(...)):
    """（已废弃，请用 /upload/videos）上传视频并同步完成识别。"""
    if direction not in ["front", "backend", "normal"]:
        return JSONResponse(status_code=400, content={"error": "direction 只能是 front/backend/normal"})
    bad = _check_upload_basic(request, file)
    if bad is not None:
        return bad
    try:
        file_path = await save_file(file, settings.VIDEO_DIR)
    except UploadTooLargeError as e:
        return JSONResponse(status_code=413, content={"error": str(e)})
    generate_video_path, generate_image_path, flip_video_path = file_name_extract(file_path)

    try:
        if direction == "backend":
            await asyncio.get_running_loop().run_in_executor(
                request.app.state.executor, rotate_video, file_path, flip_video_path, "cw")
        elif direction == "front":
            await asyncio.get_running_loop().run_in_executor(
                request.app.state.executor, rotate_video, file_path, flip_video_path, "ccw")
        else:
            flip_video_path = file_path
    except Exception as e:
        return _error_response(500, f"视频旋转失败: {e}", e)

    try:
        results = await video_tasks.run_pipeline(
            request.app.state, flip_video_path, generate_video_path, generate_image_path, fast=False
        )
        return results
    except Exception as e:
        return _error_response(500, "识别处理失败，请查看服务器日志", e)


@app.head("/upload/videos", tags=["上传地址探活（App 测试连接，不处理视频）"])
async def upload_videos_head():
    return Response(status_code=200)


@app.post("/upload/videos", tags=["上传要检测的视频(无参数版)"],
          dependencies=[Depends(verify_api_key)])
async def upload_videos_simple(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    wait: bool = Query(False, description="true=同步等待识别；false=先返回task_id(默认)"),
    fast: bool = Query(True, description="true=仅唇语VSR(更快)；false=年龄+VSR+唇部视频"),
):
    bad = _check_upload_basic(request, file)
    if bad is not None:
        return bad

    try:
        file_path = await save_file(file, settings.VIDEO_DIR)
    except UploadTooLargeError as e:
        return JSONResponse(status_code=413, content={"error": str(e)})
    except Exception as e:
        return _error_response(500, "保存文件失败", e)

    generate_video_path, generate_image_path, flip_video_path = file_name_extract(file_path)
    flip_video_path = file_path
    image_name = os.path.basename(generate_image_path)
    video_name = os.path.basename(generate_video_path)

    if not wait:
        task_id = video_tasks.create_task(flip_video_path, image_name, video_name)
        background_tasks.add_task(
            video_tasks.process_task_background,
            request.app.state,
            task_id,
            flip_video_path,
            generate_video_path,
            generate_image_path,
            fast,
        )
        return {
            "status": "processing",
            "task_id": task_id,
            "message": "文件已接收，识别处理中",
            "image": image_name,
            "video": video_name,
        }

    try:
        results = await video_tasks.run_pipeline(
            request.app.state, flip_video_path, generate_video_path, generate_image_path, fast=fast
        )
        return attach_plain_text(results)
    except Exception as e:
        return _error_response(500, "识别处理失败，请查看服务器日志", e)


@app.get("/upload/videos/status/{task_id}", tags=["查询异步上传识别结果"],
         dependencies=[Depends(verify_api_key)])
async def upload_videos_status(task_id: str):
    task = video_tasks.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if task["status"] == "processing":
        return {"status": "processing", "task_id": task_id}
    if task["status"] == "error":
        return {
            "status": "error",
            "task_id": task_id,
            "error": task.get("error"),
            "image": task.get("image", ""),
            "video": task.get("video", ""),
        }
    return {
        "status": "done",
        "task_id": task_id,
        "image": task.get("image", ""),
        "video": task.get("video", ""),
        "src_age_estimate": task.get("src_age_estimate", ""),
        "src_recognition_results": task.get("src_recognition_results", ""),
        "plain_text": task.get("plain_text") or task.get("src_recognition_results", ""),
    }


@app.get("/recognition/latest", tags=["获取最近一次唇语纯文本（家居服务拉取）"],
         dependencies=[Depends(verify_api_key)])
async def recognition_latest():
    latest = get_latest()
    return {
        "seq": latest.get("seq", 0),
        "plain_text": latest.get("plain_text", ""),
        "src_recognition_results": latest.get("src_recognition_results", ""),
        "src_age_estimate": latest.get("src_age_estimate", ""),
        "image": latest.get("image", ""),
        "video": latest.get("video", ""),
        "updated_at": latest.get("updated_at", 0),
    }


@app.get("/image/{filename}", tags=["获取检测的年龄性别图像"],
         dependencies=[Depends(verify_api_key)])
async def get_image(filename: str):
    return _serve_media(settings.GENERATE_IMAGES_DIR, filename)


@app.get("/video/{filename}", tags=["获取生成的唇部视频"],
         dependencies=[Depends(verify_api_key)])
async def get_video(filename: str):
    return _serve_media(settings.GENERATE_VIDEOS_DIR, filename)


@app.get("/test", tags=["服务测试"])
async def test():
    return {"status": "ok", "message": "服务运行正常"}


@app.get("/edit_distance", tags=["句子编辑距离得分计算"],
         dependencies=[Depends(verify_api_key)])
async def edit_distance(sentence1: str, sentence2: str):
    similarity, dist = await asyncio.to_thread(pinyin_similarity_score, sentence1, sentence2)
    return {"score": similarity, "distance": dist}


@app.post("/llm_interaction", tags=["大模型数据总结，输出错误修复"],
          dependencies=[Depends(verify_api_key)])
async def llm_interaction(request: Request, llm_interaction_model: LLMInteractionModel):
    # image_path 只允许是产物目录内的文件名，杜绝任意文件被上传到第三方
    image_path = safe_resolve(settings.GENERATE_IMAGES_DIR, llm_interaction_model.image_path)
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    session = request.app.state.http_session
    try:
        file_id = await upload_image_file(session, str(image_path))
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=502, detail=f"文件上传到 Dify 失败: {e}")
    if file_id is None:
        raise HTTPException(status_code=502, detail="文件上传到 Dify 失败")

    llm_results = await run_workflow(
        session,
        file_id,
        llm_interaction_model.src_age_estimate,
        llm_interaction_model.src_recognition_results,
    )
    if llm_results is False:
        raise HTTPException(status_code=502, detail="LLM workflow 执行失败")
    return llm_results


if __name__ == '__main__':
    uvicorn.run("app2:app", host=settings.HOST, port=settings.PORT)