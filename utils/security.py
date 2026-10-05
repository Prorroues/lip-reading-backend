# -*- coding: utf-8 -*-
"""安全相关工具：路径穿越防护 + 可选 API Key 鉴权。"""
import logging
from pathlib import Path

from fastapi import Header, HTTPException

from utils import settings

logger = logging.getLogger("backend.security")


def safe_resolve(directory: Path, filename: str) -> Path:
    """把用户传入的文件名限制在指定目录内，防止 ../../ 路径穿越。

    合法返回绝对路径；非法抛出 404（不暴露路径细节）。
    """
    base = directory.resolve()
    target = (base / filename).resolve()
    if base != target and base not in target.parents:
        logger.warning("拒绝路径穿越访问: %r", filename)
        raise HTTPException(status_code=404, detail="File not found")
    return target


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """可选鉴权：设置了 BACKEND_API_KEY 才强制校验，未设置保持原行为（放行）。

    用法：在路由上加 dependencies=[Depends(verify_api_key)]。
    """
    if not settings.BACKEND_API_KEY:
        return
    if x_api_key != settings.BACKEND_API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")