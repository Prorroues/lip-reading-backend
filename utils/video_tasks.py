import asyncio
import time
import traceback
import uuid
from typing import Any

from utils.latest_recognition import attach_plain_text, set_latest

_tasks: dict[str, dict[str, Any]] = {}
_MAX_TASKS = 200


def _prune_tasks() -> None:
    if len(_tasks) <= _MAX_TASKS:
        return
    done = [
        (tid, t.get("finished_at", 0))
        for tid, t in _tasks.items()
        if t.get("status") in ("done", "error")
    ]
    done.sort(key=lambda x: x[1])
    for tid, _ in done[: len(_tasks) - _MAX_TASKS]:
        _tasks.pop(tid, None)


def create_task(file_path: str, image_name: str, video_name: str) -> str:
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "processing",
        "file_path": file_path,
        "image": image_name,
        "video": video_name,
        "src_age_estimate": "",
        "src_recognition_results": "",
        "error": None,
        "created_at": time.time(),
        "finished_at": None,
    }
    _prune_tasks()
    return task_id


def get_task(task_id: str) -> dict[str, Any] | None:
    return _tasks.get(task_id)


def mark_done(task_id: str, results: dict[str, Any]) -> None:
    task = _tasks.get(task_id)
    if not task:
        return
    task.update(results)
    task["status"] = "done"
    task["finished_at"] = time.time()
    attach_plain_text(task)


def mark_error(task_id: str, error: str) -> None:
    task = _tasks.get(task_id)
    if not task:
        return
    task["status"] = "error"
    task["error"] = error
    task["finished_at"] = time.time()


async def run_pipeline(
    request,
    flip_video_path: str,
    generate_video_path: str,
    generate_image_path: str,
    fast: bool = False,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    results = {
        "image": generate_image_path.split("/")[-1],
        "video": generate_video_path.split("/")[-1],
        "src_age_estimate": "",
        "src_recognition_results": "",
    }

    async def run_vsr():
        try:
            results["src_recognition_results"] = await loop.run_in_executor(
                request.app.state.executor,
                request.app.state.vsr.process_videos,
                flip_video_path,
            )
        except Exception as e:
            results["src_recognition_results"] = f"错误: {str(e)}"
            print(f"process_videos 错误: {traceback.format_exc()}")

    if fast:
        await run_vsr()
        attach_plain_text(results)
        set_latest(results)
        return results

    async with asyncio.TaskGroup() as tg:

        async def run_prediction():
            try:
                results["src_age_estimate"] = await loop.run_in_executor(
                    request.app.state.executor,
                    request.app.state.age_estimate.extract_frames,
                    flip_video_path,
                    generate_image_path,
                )
            except Exception as e:
                results["src_age_estimate"] = f"错误: {str(e)}"
                print(f"extract_frames 错误: {traceback.format_exc()}")

        async def run_lip_video():
            try:
                await loop.run_in_executor(
                    request.app.state.executor,
                    request.app.state.lip_video.infer,
                    flip_video_path,
                    generate_video_path,
                )
            except Exception:
                print(f"lip_video.infer 错误: {traceback.format_exc()}")

        tg.create_task(run_prediction())
        tg.create_task(run_vsr())
        tg.create_task(run_lip_video())

    attach_plain_text(results)
    set_latest(results)
    return results


async def process_task_background(
    request,
    task_id: str,
    flip_video_path: str,
    generate_video_path: str,
    generate_image_path: str,
    fast: bool = False,
) -> None:
    try:
        results = await run_pipeline(
            request, flip_video_path, generate_video_path, generate_image_path, fast=fast
        )
        mark_done(task_id, results)
    except Exception as e:
        mark_error(task_id, str(e))
        print(f"后台任务失败 task={task_id}: {traceback.format_exc()}")
