# -*- coding: utf-8 -*-
"""Dify 文件上传与工作流调用。

优化点：
- 端口/地址/密钥统一走 utils.settings，不再硬编码 21000
- 复用 lifespan 创建的全局 aiohttp session（连接池），不再每次新建
- 上传改为流式（不再把整文件读进内存）
- 非 JSON 响应、网络错误都有明确报错
"""
import logging
import mimetypes
import os

import aiohttp

from utils import settings

logger = logging.getLogger("backend.dify")


def _dify_url(path: str) -> str:
    settings.require_dify_config()
    return f"http://{settings.DIFY_BASE_URL}:{settings.DIFY_PORT}{path}"


async def upload_image_file(session: aiohttp.ClientSession, file_path: str):
    """流式上传文件到 Dify，成功返回 file_id，失败抛异常。"""
    upload_url = _dify_url("/v1/files/upload")
    headers = {"Authorization": f"Bearer {settings.DIFY_API_KEY}"}

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    mime_type, _ = mimetypes.guess_type(file_path)
    file_name = os.path.basename(file_path)
    timeout = aiohttp.ClientTimeout(total=120)

    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field(
            name="file",
            value=f,  # 传文件对象，aiohttp 流式发送
            filename=file_name,
            content_type=mime_type or "application/octet-stream",
        )
        data.add_field("user", settings.DIFY_APP_USER)

        try:
            async with session.post(upload_url, headers=headers, data=data, timeout=timeout) as response:
                try:
                    text = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    text = {"raw": (await response.text())[:500]}
                if response.status in (200, 201):
                    return text.get("id")
                logger.error("Dify 上传失败 status=%s body=%s", response.status, text)
                raise RuntimeError(f"Dify 上传失败: {response.status} - {text}")
        except aiohttp.ClientError as e:
            logger.error("Dify 上传网络错误 %s: %s", upload_url, e)
            raise


async def run_workflow(session: aiohttp.ClientSession, file_id, src_age_estimate,
                       src_recognition_results, response_mode="blocking"):
    workflow_url = _dify_url("/v1/workflows/run")
    headers = {
        "Authorization": f"Bearer {settings.DIFY_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "inputs": {
            "image_upload": {
                "transfer_method": "local_file",
                "upload_file_id": file_id,
                "type": "image",
            },
            "src_age_estimate": src_age_estimate,
            "src_recognition_results": src_recognition_results,
        },
        "response_mode": response_mode,
        "user": settings.DIFY_APP_USER,
    }
    timeout = aiohttp.ClientTimeout(total=180)

    try:
        logger.info("运行 Dify 工作流...")
        async with session.post(workflow_url, headers=headers, json=data, timeout=timeout) as response:
            try:
                resp_json = await response.json()
            except (aiohttp.ContentTypeError, ValueError):
                resp_json = {"raw": (await response.text())[:500]}
            if response.status == 200:
                logger.info("Dify 工作流执行成功")
                return resp_json["data"]["outputs"]
            logger.error("Dify 工作流失败 status=%s body=%s", response.status, resp_json)
            return False
    except (aiohttp.ClientError, KeyError) as e:
        logger.exception("Dify 工作流调用异常: %s", e)
        return False