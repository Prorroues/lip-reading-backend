# -*- coding: utf-8 -*-
"""对外按路径切分：/home/* 反代到家居 FastAPI（127.0.0.1:8000）。"""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from utils import settings

logger = logging.getLogger("backend.gateway")

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _upstream_path(full_path: str) -> str:
    """/home 或 /home/ -> / ；/home/api/x -> /api/x"""
    path = (full_path or "").lstrip("/")
    return f"/{path}" if path else "/"


async def proxy_to_home(request: Request, path: str):
    client: httpx.AsyncClient = request.app.state.home_http
    url = path if path.startswith("/") else "/" + path
    if request.url.query:
        url = f"{url}?{request.url.query}"
    body = await request.body()
    req = client.build_request(
        request.method,
        url,
        headers=_filter_headers(request.headers),
        content=body or None,
    )
    try:
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as e:
        logger.error("反代家居失败 %s %s: %s", request.method, url, e)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=502, content={"error": "家居服务不可用", "detail": str(e)})

    out_headers = _filter_headers(upstream.headers)
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=out_headers,
        background=BackgroundTask(upstream.aclose),
    )


async def proxy_home_websocket(client_ws: WebSocket) -> None:
    await client_ws.accept()
    upstream_base = settings.HOME_UPSTREAM
    ws_url = upstream_base.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    try:
        import websockets
    except ImportError:
        logger.error("缺少 websockets 库，无法反代 /home/ws")
        await client_ws.close(code=1011)
        return

    try:
        async with websockets.connect(ws_url, open_timeout=5) as server_ws:
            async def client_to_server():
                try:
                    while True:
                        msg = await client_ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if msg.get("text") is not None:
                            await server_ws.send(msg["text"])
                        elif msg.get("bytes") is not None:
                            await server_ws.send(msg["bytes"])
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass

            async def server_to_client():
                try:
                    async for message in server_ws:
                        if isinstance(message, bytes):
                            await client_ws.send_bytes(message)
                        else:
                            await client_ws.send_text(message)
                except Exception:
                    pass

            await asyncio.gather(client_to_server(), server_to_client())
    except Exception as e:
        logger.warning("WebSocket 反代失败: %s", e)
        try:
            await client_ws.close()
        except Exception:
            pass


def register_home_proxy(app: FastAPI) -> None:
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]

    @app.api_route("/home", methods=methods, include_in_schema=False)
    @app.api_route("/home/{full_path:path}", methods=methods, include_in_schema=False)
    async def proxy_home(request: Request, full_path: str = ""):
        return await proxy_to_home(request, _upstream_path(full_path))

    @app.websocket("/home/ws")
    async def proxy_ws(ws: WebSocket):
        await proxy_home_websocket(ws)
