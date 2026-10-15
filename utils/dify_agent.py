# -*- coding: utf-8 -*-
"""调用 Dify Agent（/v1/chat-messages），与纠错工作流分开。"""

import json
import logging

import aiohttp

from utils import settings

logger = logging.getLogger("backend.dify_agent")


def _chat_url() -> str:
    if not settings.DIFY_BASE_URL:
        raise RuntimeError("缺少环境变量 base_url")
    return f"http://{settings.DIFY_BASE_URL}:{settings.DIFY_PORT}/v1/chat-messages"


async def run_assistant_chat(session: aiohttp.ClientSession, text: str) -> str:
    api_key = settings.DIFY_ASSISTANT_API_KEY
    if not api_key:
        raise RuntimeError("未配置 DIFY_ASSISTANT_API_KEY / api_key")
    query = (text or "").strip()
    if not query:
        raise ValueError("text 为空")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "inputs": {},
        "response_mode": "streaming",
        "user": settings.DIFY_APP_USER,
    }
    timeout = aiohttp.ClientTimeout(total=180)
    logger.info("调用 Dify Agent: %s", query[:80])
    async with session.post(_chat_url(), headers=headers, json=payload, timeout=timeout) as response:
        raw = await response.text()
        if response.status != 200:
            logger.error("Dify Agent 失败 status=%s body=%s", response.status, raw[:500])
            raise RuntimeError(f"Dify Agent 失败: {response.status} - {raw[:500]}")
        answer_parts: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            piece = event.get("answer") or ""
            if kind in ("agent_message", "message") and piece:
                answer_parts.append(str(piece))
        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError(f"Dify Agent 返回空回复: {raw[:300]}")
        return answer
