"""保存最近一次唇语识别纯文本，供家居服务从 24060 拉取。

不修改原有字段含义，只额外记录一份 plain_text。
"""

from __future__ import annotations

import time
from typing import Any

_latest: dict[str, Any] = {
    "seq": 0,
    "plain_text": "",
    "src_recognition_results": "",
    "src_age_estimate": "",
    "image": "",
    "video": "",
    "updated_at": 0.0,
}


def attach_plain_text(payload: dict[str, Any]) -> dict[str, Any]:
    """在原有返回上增加 plain_text，不删除已有字段。"""
    text = payload.get("src_recognition_results") or ""
    payload["plain_text"] = text
    return payload


def set_latest(results: dict[str, Any]) -> None:
    text = (results.get("src_recognition_results") or results.get("plain_text") or "").strip()
    _latest["seq"] = int(_latest.get("seq") or 0) + 1
    _latest["plain_text"] = text
    _latest["src_recognition_results"] = results.get("src_recognition_results") or text
    _latest["src_age_estimate"] = results.get("src_age_estimate") or ""
    _latest["image"] = results.get("image") or ""
    _latest["video"] = results.get("video") or ""
    _latest["updated_at"] = time.time()


def get_latest() -> dict[str, Any]:
    return dict(_latest)
