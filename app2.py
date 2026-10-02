from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import mimetypes
from pathlib import Path
import asyncio
import sys
import os
import uvicorn
import traceback
from pydantic_models.request_models import LLMInteractionModel
from utils.dify_upload import upload_image_file, run_workflow
from utils.file_handler import save_file
from utils.rotate_videos import rotate_video_counterclockwise_sync, rotate_video_clockwise_sync
from utils.file_name_extract import file_name_extract
from utils.edit_distance import pinyin_similarity_score
from utils import video_tasks
from utils.latest_recognition import attach_plain_text, get_latest, set_latest
from middlewares.init_lifespan import tai_middleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Visual_Speech_Recognition_for_Multiple_Languages')))
app = FastAPI(lifespan=tai_middleware)

# 添加CORS中间件，允许所有来源访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)
UPLOAD_DIR = Path("uploads")
VIDEO_DIR = UPLOAD_DIR / "videos"

GENERATE_DIR = Path("generates")
GENERATE_VIDEOS_DIR = GENERATE_DIR / "videos"
GENERATE_IMAGES_DIR = GENERATE_DIR / "images"


@app.post("/upload/video", tags=["上传要检测的视频"])
async def upload_video(request: Request, file: UploadFile = File(...), direction: str = Form(...)):
    if not file.content_type.startswith("video/") or direction not in ["front", "backend", "normal"]:
        return JSONResponse(status_code=400, content={"error": "文件不是视频"})
    loop = asyncio.get_running_loop()
    file_path = await save_file(file, VIDEO_DIR)
    generate_video_path, generate_image_path, flip_video_path = file_name_extract(file_path)
    if direction == "backend":
        await loop.run_in_executor(
            request.app.state.executor,
            rotate_video_clockwise_sync,
            file_path,
            flip_video_path
        )
    elif direction == "front":
        await loop.run_in_executor(
            request.app.state.executor,
            rotate_video_counterclockwise_sync,
            file_path,
            flip_video_path
        )
    else:
        flip_video_path = file_path

    results = {
        "image": os.path.basename(generate_image_path),
        "video": os.path.basename(generate_video_path),
        "src_age_estimate": "",
        "src_recognition_results": "",
    }

    # 并发执行 extract_frames 和 vsr.process_videos
    async with asyncio.TaskGroup() as tg:

        async def run_prediction():
            result = await loop.run_in_executor(
                request.app.state.executor,
                request.app.state.age_estimate.extract_frames,
                flip_video_path,
                generate_image_path
            )
            results["src_age_estimate"] = result

        async def run_vsr():
            result = await loop.run_in_executor(
                request.app.state.executor,
                request.app.state.vsr.process_videos,
                flip_video_path,
            )
            results["src_recognition_results"] = result

        async def run_lip_video():
            await loop.run_in_executor(
                request.app.state.executor,
                request.app.state.lip_video.infer,
                flip_video_path,
                generate_video_path
            )


        tg.create_task(run_prediction())
        tg.create_task(run_vsr())
        tg.create_task(run_lip_video())

    attach_plain_text(results)
    set_latest(results)
    return results


@app.head("/upload/videos", tags=["上传地址探活（App 测试连接，不处理视频）"])
async def upload_videos_head():
    return JSONResponse(status_code=200, content={"status": "ok", "path": "/upload/videos"})


@app.post("/upload/videos", tags=["上传要检测的视频(无参数版)"])
async def upload_videos_simple(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    wait: bool = Query(False, description="true=同步等待识别；false=先返回task_id(默认)"),
    fast: bool = Query(True, description="true=仅唇语VSR(更快)；false=年龄+VSR+唇部视频"),
):
    try:
        if not file.content_type or not file.content_type.startswith("video/"):
            return JSONResponse(
                status_code=400,
                content={"error": f"文件不是视频类型，收到类型: {file.content_type}", "filename": file.filename}
            )

        if file.filename:
            filename_lower = file.filename.lower()
            valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
            if not any(filename_lower.endswith(ext) for ext in valid_extensions):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"不支持的文件格式: {file.filename}，只支持: {', '.join(valid_extensions)}"}
                )

        try:
            file_path = await save_file(file, VIDEO_DIR)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"保存文件失败: {str(e)}", "traceback": traceback.format_exc()}
            )

        try:
            generate_video_path, generate_image_path, flip_video_path = file_name_extract(file_path)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"提取文件名路径失败: {str(e)}", "traceback": traceback.format_exc()}
            )

        flip_video_path = file_path
        image_name = os.path.basename(generate_image_path)
        video_name = os.path.basename(generate_video_path)

        if not wait:
            task_id = video_tasks.create_task(flip_video_path, image_name, video_name)
            background_tasks.add_task(
                video_tasks.process_task_background,
                request,
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

        results = await video_tasks.run_pipeline(
            request, flip_video_path, generate_video_path, generate_image_path, fast=fast
        )
        return attach_plain_text(results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"服务器内部错误: {str(e)}", "traceback": traceback.format_exc()}
        )


@app.get("/upload/videos/status/{task_id}", tags=["查询异步上传识别结果"])
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


@app.get("/recognition/latest", tags=["获取最近一次唇语纯文本（家居服务拉取）"])
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


@app.get("/image/{filename}", tags=["获取检测的年龄性别图像"])
async def get_media(filename: str):
    file_path = GENERATE_IMAGES_DIR / filename

    # Check if file exists
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    media_type, _ = mimetypes.guess_type(file_path)
    if not media_type:
        media_type = "application/octet-stream"

    # Return file response
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@app.get("/video/{filename}", tags=["获取生成的唇部视频"])
async def get_media(filename: str):
    file_path = GENERATE_VIDEOS_DIR / filename

    # Check if file exists
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    media_type, _ = mimetypes.guess_type(file_path)
    if not media_type:
        media_type = "application/octet-stream"

    # Return file response
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@app.get("/test", tags=["服务测试"])
async def test():
    return {"status": "ok", "message": "服务运行正常"}

@app.get("/edit_distance", tags=["句子编辑距离得分计算"])
async def edit_distance(sentence1: str, sentence2: str):
    distance = await asyncio.to_thread(pinyin_similarity_score, sentence1, sentence2)
    return {"score": distance}

@app.post("/llm_interaction", tags=["大模型数据总结，输出错误修复"])
async def llm_interaction(request: Request, llm_interaction_model: LLMInteractionModel):
    file_id = await upload_image_file(f"generates/images/{llm_interaction_model.image_path}")
    if file_id is None:
        raise HTTPException(status_code=404, detail="File not found or Upload Failed")
    llm_results = await run_workflow(file_id,
                                     llm_interaction_model.src_age_estimate,
                                     llm_interaction_model.src_recognition_results)
    if llm_results is False:
        raise HTTPException(status_code=404, detail="LLM failed")
    return llm_results




if __name__ == '__main__':
    uvicorn.run("app2:app", host='0.0.0.0', port=24060)
    #  outport 41399

