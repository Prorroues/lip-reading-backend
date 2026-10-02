"""Backup of closed-vocab home CTC decoder (saved 2026-10-01).

Restore by copying this file over command_decode.py, then in model.py
return "未匹配到家居指令" when pick_command has no match (do not beam-search).

Closed-vocab home commands scored by visual CTC.
Always pick from ACTIVE_COMMANDS. Do not emit greedy/news text.

Scoring notes:
- Do NOT divide CTC NLL by sqrt(len). When the mouth is unclear, that
  formula always prefers the longest phrase, which is 开/关床下前灯.
  That is why 「打开风扇」 kept coming back as 床下前灯, and why 开/关
  flipped between trials (those two visemes are almost the same).
- Pick the device first, then open vs close on same-length pairs
  (打开风扇 vs 关闭风扇). Do not score 开风扇/关风扇; the shorter
  关风扇 always beats 打开风扇 and then gets rewritten as 关闭风扇.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn.functional as F

logger = logging.getLogger("command_decode")

FAN_ON = ("打开风扇",)
FAN_OFF = ("关闭风扇",)
LIGHT_ON = (
    "开厕所灯",
    "开床后灯",
    "开床下前灯",
    "开床下灯",
    "开上左灯",
)
LIGHT_OFF = (
    "关厕所灯",
    "关床后灯",
    "关床下前灯",
    "关床下灯",
    "关上左灯",
)
ACTIVE_COMMANDS: tuple[str, ...] = FAN_ON + FAN_OFF + LIGHT_ON + LIGHT_OFF

DEVICE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("风扇", FAN_ON + FAN_OFF),
    ("厕所灯", ("开厕所灯", "关厕所灯")),
    ("床后灯", ("开床后灯", "关床后灯")),
    ("床下前灯", ("开床下前灯", "开床下灯", "关床下前灯", "关床下灯")),
    ("上左灯", ("开上左灯", "关上左灯")),
)

CANONICAL = {
    "开风扇": "打开风扇",
    "打开风扇": "打开风扇",
    "开启风扇": "打开风扇",
    "启动风扇": "打开风扇",
    "关风扇": "关闭风扇",
    "关闭风扇": "关闭风扇",
    "关掉风扇": "关闭风扇",
    "停止风扇": "关闭风扇",
    "停风扇": "关闭风扇",
    "开厕所灯": "打开厕所灯",
    "关厕所灯": "关闭厕所灯",
    "开床后灯": "打开床后灯",
    "关床后灯": "关闭床后灯",
    "开床下灯": "打开床下前灯",
    "开床下前灯": "打开床下前灯",
    "关床下灯": "关闭床下前灯",
    "关床下前灯": "关闭床下前灯",
    "开上左灯": "打开上左灯",
    "关上左灯": "关闭上左灯",
}

BLANK_ID = 0
OPEN_HINTS = ("打", "启")
CLOSE_HINTS = ("关", "停", "闭")
ACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("打开风扇", "关闭风扇"),
    ("开厕所灯", "关厕所灯"),
    ("开床后灯", "关床后灯"),
    ("开床下前灯", "关床下前灯"),
    ("开上左灯", "关上左灯"),
)
DEVICE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("风扇", ("风", "扇")),
    ("厕所灯", ("厕",)),
    ("床后灯", ("后",)),
    ("床下前灯", ("下", "前")),
    ("上左灯", ("左",)),
)


def _char_ids(phrase: str, token_list: list[str]) -> list[int] | None:
    ids: list[int] = []
    for ch in phrase:
        try:
            idx = token_list.index(ch)
        except ValueError:
            return None
        if idx == BLANK_ID:
            return None
        ids.append(idx)
    return ids or None


def ctc_greedy_text(logp: torch.Tensor, token_list: list[str]) -> str:
    ids = logp.argmax(dim=-1).tolist()
    chars: list[str] = []
    prev = None
    for i in ids:
        if i == BLANK_ID:
            prev = BLANK_ID
            continue
        if i == prev:
            continue
        prev = i
        if 0 <= i < len(token_list):
            tok = token_list[i]
            if tok and not tok.startswith("<"):
                chars.append(tok)
    return "".join(chars)


def score_commands(
    enc_feats: torch.Tensor,
    ctc_module,
    token_list: list[str],
    greedy: str,
    commands: Iterable[str] = ACTIVE_COMMANDS,
) -> list[tuple[float, str, float]]:
    """Return (score, phrase, raw_nll) sorted ascending. Lower is better."""
    if enc_feats.dim() != 2:
        raise ValueError(f"expected encoder (T, D), got {tuple(enc_feats.shape)}")
    logp = ctc_module.log_softmax(enc_feats.unsqueeze(0))
    tlen = logp.size(1)
    logp_t = logp.permute(1, 0, 2).float().cpu()
    scored: list[tuple[float, str, float]] = []
    distinctive = set()
    for _, hints in DEVICE_HINTS:
        distinctive.update(hints)
    for phrase in commands:
        ids = _char_ids(phrase, token_list)
        if not ids:
            continue
        targets = torch.tensor(ids, dtype=torch.long)
        loss = F.ctc_loss(
            logp_t,
            targets,
            torch.tensor([tlen], dtype=torch.long),
            torch.tensor([len(ids)], dtype=torch.long),
            blank=BLANK_ID,
            reduction="sum",
            zero_infinity=True,
        )
        raw = float(loss.item())
        hits = sum(1 for ch in phrase if ch in greedy and ch in distinctive) if greedy else 0
        action_hits = 0
        if greedy and 1 <= len(greedy) <= 6:
            is_off = phrase in FAN_OFF or phrase in LIGHT_OFF
            is_on = phrase in FAN_ON or phrase in LIGHT_ON
            if is_off and any(ch in greedy for ch in CLOSE_HINTS):
                action_hits += 1
            if is_on and any(ch in greedy for ch in OPEN_HINTS):
                action_hits += 1
        score = raw - 8.0 * hits - 8.0 * action_hits
        scored.append((score, phrase, raw))
    scored.sort(key=lambda x: x[0])
    return scored


def _hinted_device(greedy: str) -> str | None:
    if not greedy:
        return None
    hits = [
        name
        for name, chars in DEVICE_HINTS
        if any(ch in greedy for ch in chars)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _pick_open_close(
    greedy: str,
    rows: list[tuple[float, str, float]],
) -> tuple[float, str, float]:
    by_phrase = {row[1]: row for row in rows}
    choices: list[tuple[float, str, float]] = []
    for on_p, off_p in ACTION_PAIRS:
        on_row = by_phrase.get(on_p)
        off_row = by_phrase.get(off_p)
        if on_row is None or off_row is None:
            continue
        # 开/关口型太像，关闭必须明显更像才算关，否则保持打开。
        need = 2.5
        has_close = bool(greedy) and 1 <= len(greedy) <= 6 and any(
            ch in greedy for ch in CLOSE_HINTS
        )
        has_open = bool(greedy) and any(ch in greedy for ch in OPEN_HINTS)
        if off_row[0] + need < on_row[0] or (has_close and not has_open and off_row[0] <= on_row[0]):
            choices.append(off_row)
        else:
            choices.append(on_row)
    if choices:
        choices.sort(key=lambda x: x[0])
        return choices[0]
    return rows[0]


def pick_command(
    enc_feats: torch.Tensor,
    ctc_module,
    token_list: list[str],
) -> tuple[str | None, str, list[tuple[float, str]]]:
    logp = ctc_module.log_softmax(enc_feats.unsqueeze(0)).squeeze(0)
    greedy = ctc_greedy_text(logp, token_list)
    ranked = score_commands(enc_feats, ctc_module, token_list, greedy, ACTIVE_COMMANDS)
    if not ranked:
        logger.info("CTC greedy=%r empty candidate table", greedy)
        return None, greedy, []

    by_device: list[tuple[float, str, list[tuple[float, str, float]]]] = []
    for name, phrases in DEVICE_GROUPS:
        rows = [row for row in ranked if row[1] in phrases]
        if rows:
            by_device.append((rows[0][0], name, rows))
    by_device.sort(key=lambda x: x[0])

    hinted = _hinted_device(greedy)
    if hinted and len(greedy) <= 8:
        hinted_item = next((item for item in by_device if item[1] == hinted), None)
        top_score = by_device[0][0]
        if hinted_item is not None and hinted_item[0] <= top_score + max(6.0, 0.08 * abs(top_score)):
            by_device = [hinted_item] + [item for item in by_device if item[1] != hinted]

    device_score, device_name, rows = by_device[0]
    second = by_device[1][0] if len(by_device) > 1 else device_score + 1.0
    best = _pick_open_close(greedy, rows)
    phrase = CANONICAL.get(best[1], best[1])
    logger.info(
        "closed-vocab greedy=%r device=%s best=%s raw=%.3f score=%.3f "
        "device_margin=%.3f top3=%s",
        greedy,
        device_name,
        phrase,
        best[2],
        best[0],
        second - device_score,
        [(round(s, 2), CANONICAL.get(p, p)) for s, p, _ in ranked[:3]],
    )
    return phrase, greedy, [(s, CANONICAL.get(p, p)) for s, p, _ in ranked]
