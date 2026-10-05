# -*- coding: utf-8 -*-
"""拼音相似度计算：把两个中文句子转成拼音后算编辑距离与相似度。

返回语义（修正版）：
    (similarity, edit_distance)
    similarity     0~1 的相似度，越大越像，= 1 - 编辑距离 / 较长拼音串长度
    edit_distance  两条拼音串之间的真实编辑距离（旧版返回值语义混乱，已修正）
"""
from typing import Tuple

from Levenshtein import distance
from pypinyin import Style, pinyin

# 易混淆音归一化表：前后鼻音、l/n 不分
_SOUND_MAP = (
    ("in", "ing"),
    ("en", "eng"),
    ("an", "ang"),
    ("un", "ung"),
    ("l", "n"),
)


def normalize_pinyin(pinyin_str: str) -> str:
    """对拼音串做易混淆音归一化（in/ing、en/eng、an/ang、un/ung、l/n）。"""
    for src, dst in _SOUND_MAP:
        pinyin_str = pinyin_str.replace(src, dst)
    return pinyin_str


def pinyin_similarity_score(sentence1: str, sentence2: str) -> Tuple[float, int]:
    """计算两个中文句子的拼音相似度与编辑距离。

    Args:
        sentence1: 第一个中文句子
        sentence2: 第二个中文句子

    Returns:
        (similarity, edit_distance)：相似度 0~1（越大越相似）和真实编辑距离
    """
    pinyin1 = "".join(item[0] for item in pinyin(sentence1, style=Style.NORMAL))
    pinyin2 = "".join(item[0] for item in pinyin(sentence2, style=Style.NORMAL))

    p1 = normalize_pinyin(pinyin1)
    p2 = normalize_pinyin(pinyin2)

    edit_dist = distance(p1, p2)
    max_len = max(len(p1), len(p2))
    similarity = 1.0 if max_len == 0 else max(0.0, 1.0 - edit_dist / max_len)

    return similarity, edit_dist


if __name__ == "__main__":
    s1 = "你好世界"
    s2 = "你好地球"
    score, dist = pinyin_similarity_score(s1, s2)
    print(f"Sentences: '{s1}' and '{s2}'")
    print(f"Similarity score: {score:.4f}")
    print(f"Edit distance: {dist}")