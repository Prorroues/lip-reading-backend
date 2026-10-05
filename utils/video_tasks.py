# -*- coding: utf-8 -*-
"""异步任务管理 + 推理流水线。

优化点：
- 任务状态带 TTL：完成的任务 1 小时（可配）后清理；卡死的 processing 任务
  超过 TASK_TIMEOUT_SECONDS 自动判 error，不再永远"处理中"
- 推理调用统一过信号量限并发，避免 GPU/NPU 显存互抢
- 后台任务只接收 app.state，不再传整个 request 对象
- VSR 失败直接判任务失败，不再把错误文本当识别结果推给家居系统
"""
import asyncio
import logging
import os
import time
import uuid
from typing import Any

from utils import settings
from utils.latest_recognition import attach_plain_text, set_latest

logger = logging.getLogger("backend.video_tasks")

_tasks: dict[str, dict[str, Any]] = {}
_MAX_TASKS = 200


def _prune_tasks() -> None:
    now = time.time()
    # 1) 过期的已完成/失败任务按 TTL 清理
    expired = [
        tid for tid, t in _tasks.items()
        if t.get("status") in ("done", "error")
        and now - (t.get("finished_at") or 0) > settings.TASK_TTL_SECONDS
    ]
    for tid in expired:
        _tasks.pop(tid, None)
    # 2) 数量仍超上限时，按完成时间从旧到新删
    if len(_tasks) > _MAX_TASKS:
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
    task = _tasks.get(task_id)
    # 卡死任务超时判失败
    if task and task.get("status") == "processing":
        if time.time() - task.get("created_at", 0) > settings.TASK_TIMEOUT_SECONDS:
            mark_error(task_id, "任务处理超时")
            task = _tasks.get(task_id)
    return task


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
    state,
    flip_video_path: str,
    generate_video_path: str,
    generate_image_path: str,
    fast: bool = False,
) -> dict[str, Any]:
    """执行识别流水线。VSR 失败会抛异常（由调用方决定 500/任务失败）。

    state: request.app.state，需要上面有 executor / vsr / age_estimate /
    lip_video / infer_sem。
    """
    loop = asyncio.get_running_loop()
    results = {
        "image": os.path.basename(generate_image_path),
        "video": os.path.basename(generate_video_path),
        "src_age_estimate": "",
        "src_recognition_results": "",
    }

    async def run_vsr() -> str:
        # VSR 是核心结果：异常直接向上抛，不写进 results
        async with state.infer_sem:
            vsr_text = await loop.run_in_executor(
                state.executor, state.vsr.process_videos, flip_video_path
            )
        vsr_text = (vsr_text or "").strip()
        # 空结果保留旧版提示文案（给 APP 看），但不会写入家居系统（见下方 set_latest 条件）
        results["src_recognition_results"] = vsr_text or "唇语识别结果为空"
        return vsr_text

    if fast:
        vsr_text = await run_vsr()
        attach_plain_text(results)
        if vsr_text:
            set_latest(results)
        return results

    async def run_prediction():
        try:
            async with state.infer_sem:
                results["src_age_estimate"] = await loop.run_in_executor(
                    state.executor,
                    state.age_estimate.extract_frames,
                    flip_video_path,
                    generate_image_path,
                )
        except Exception as e:
            # 辅助模块失败降级为空，不影响主流程
            logger.exception("extract_frames 失败: %s", e)

    async def run_lip_video():
        try:
            async with state.infer_sem:
                await loop.run_in_executor(
                    state.executor,
                    state.lip_video.infer,
                    flip_video_path,
                    generate_video_path,
                )
        except Exception:
            logger.exception("lip_video.infer 失败")

    # VSR 单独一个任务：失败时干净地向上抛（不用 TaskGroup，避免 ExceptionGroup 包装）
    vsr_task = asyncio.create_task(run_vsr())
    # 辅助任务内部已捕获自身异常，gather 不会失败
    others = asyncio.gather(run_prediction(), run_lip_video())
    try:
        vsr_text = await vsr_task
    finally:
        await others  # 无论 VSR 成败，都等辅助任务收尾，不留悬挂协程

    attach_plain_text(results)
    if vsr_text:
        set_latest(results)
    return results


async def process_task_background(
    state,
    task_id: str,
    flip_video_path: str,
    generate_video_path: str,
    generate_image_path: str,
    fast: bool = False,
) -> None:
    try:
        results = await run_pipeline(
            state, flip_video_path, generate_video_path, generate_image_path, fast=fast
        )
        mark_done(task_id, results)
    except Exception as e:
        logger.exception("后台任务失败 task=%s", task_id)
        mark_error(task_id, str(e))