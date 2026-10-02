"""Live overlap-gated decoder restored 2026-10-01.

Home commands if CTC overlap is confident; otherwise infer() falls back
to beam search (open vocab / news LM). Closed-vocab snapshot is
command_decode_closedvocab.py.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn.functional as F

logger = logging.getLogger("command_decode")

FAN_ON = ("开风扇", "打开风扇", "开启风扇", "启动风扇")
FAN_OFF = ("关风扇", "关闭风扇", "关掉风扇", "停止风扇", "停风扇")
LIGHT_ON = (
    "开氛围灯",
    "开柜子灯",
    "开床头灯",
    "开床后灯",
    "开床下灯",
    "开上左灯",
    "开厕所灯",
)
LIGHT_OFF = (
    "关氛围灯",
    "关柜子灯",
    "关床头灯",
    "关床后灯",
    "关床下灯",
    "关上左灯",
    "关厕所灯",
)
SCENE = ("我回来了", "我出门了")
HOME_COMMANDS: tuple[str, ...] = FAN_ON + FAN_OFF + LIGHT_ON + LIGHT_OFF + SCENE

BLANK_ID = 0
MAX_NLL = 40.0
MIN_MARGIN = 0.3
FAN_HINTS = ("风", "扇")
LIGHT_HINTS = ("灯", "光")
OPEN_HINTS = ("开", "打", "启")
CLOSE_HINTS = ("关", "停")


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


def _overlap_bonus(phrase: str, greedy: str) -> float:
    if not greedy:
        return 0.0
    hits = sum(1 for ch in phrase if ch in greedy)
    return float(hits)


def score_commands(
    enc_feats: torch.Tensor,
    ctc_module,
    token_list: list[str],
    greedy: str,
    commands: Iterable[str] = HOME_COMMANDS,
) -> list[tuple[float, str, float]]:
    if enc_feats.dim() != 2:
        raise ValueError(f"expected encoder (T, D), got {tuple(enc_feats.shape)}")
    logp = ctc_module.log_softmax(enc_feats.unsqueeze(0))
    tlen = logp.size(1)
    logp_t = logp.permute(1, 0, 2).float().cpu()
    scored: list[tuple[float, str, float]] = []
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
        hits = _overlap_bonus(phrase, greedy)
        score = raw / (len(ids) ** 0.5) - 3.0 * hits
        scored.append((score, phrase, raw))
    scored.sort(key=lambda x: x[0])
    return scored


def _enough_overlap(phrase: str, greedy: str) -> bool:
    if not greedy:
        return False
    hits = sum(1 for ch in phrase if ch in greedy)
    if any(h in phrase and h in greedy for h in FAN_HINTS + LIGHT_HINTS):
        return hits >= 1
    return hits >= 2


def _restrict(greedy: str, ranked: list[tuple[float, str, float]]) -> list[tuple[float, str, float]]:
    ranked = [row for row in ranked if _enough_overlap(row[1], greedy)]
    if any(ch in greedy for ch in FAN_HINTS):
        fan = [row for row in ranked if row[1] in FAN_ON + FAN_OFF]
        if fan:
            ranked = fan
    elif any(ch in greedy for ch in LIGHT_HINTS):
        lights = [row for row in ranked if row[1] in LIGHT_ON + LIGHT_OFF]
        if lights:
            ranked = lights
    if any(ch in greedy for ch in OPEN_HINTS) and not any(ch in greedy for ch in CLOSE_HINTS):
        opened = [row for row in ranked if row[1] in FAN_ON + LIGHT_ON]
        if opened:
            ranked = opened
    if any(ch in greedy for ch in CLOSE_HINTS):
        closed = [row for row in ranked if row[1] in FAN_OFF + LIGHT_OFF]
        if closed:
            ranked = closed
    return ranked


def pick_command(
    enc_feats: torch.Tensor,
    ctc_module,
    token_list: list[str],
) -> tuple[str | None, str, list[tuple[float, str]]]:
    logp = ctc_module.log_softmax(enc_feats.unsqueeze(0)).squeeze(0)
    greedy = ctc_greedy_text(logp, token_list)
    news_markers = ("习近平", "习近", "总书记", "新闻联播", "人民日报")
    if any(w in greedy for w in news_markers) or (
        greedy.count("的") >= 3 and len(set(greedy)) <= 4
    ):
        logger.info("greedy looks like news/garbage, skip command pick: %r", greedy)
        return None, greedy, []

    ranked_full = score_commands(enc_feats, ctc_module, token_list, greedy)
    ranked = _restrict(greedy, ranked_full)
    if not ranked:
        logger.info("CTC greedy=%r no overlapping command", greedy)
        return None, greedy, []

    best_score, best, best_nll = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else best_score + 1.0
    margin = second_score - best_score
    logger.info(
        "CTC greedy=%r best=%s nll=%.3f score=%.3f margin=%.3f top3=%s",
        greedy,
        best,
        best_nll,
        best_score,
        margin,
        [(round(s, 2), p) for s, p, _ in ranked[:3]],
    )
    if best_nll <= MAX_NLL and margin >= MIN_MARGIN:
        return best, greedy, [(s, p) for s, p, _ in ranked]
    return None, greedy, [(s, p) for s, p, _ in ranked]
