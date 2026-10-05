# -*- coding: utf-8 -*-
"""纯逻辑单元测试：不依赖模型/GPU，开发机上跑 `pytest tests/` 即可。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import settings                      # noqa: E402
from utils.edit_distance import (               # noqa: E402
    normalize_pinyin, pinyin_similarity_score)
from utils.file_name_extract import file_name_extract  # noqa: E402
from utils.security import safe_resolve         # noqa: E402


class TestFileNameExtract:
    def test_returns_three_paths(self):
        video_path, image_path, flip_path = file_name_extract("uploads/videos/abc123.mp4")
        assert "abc123" in video_path and video_path.endswith(".mp4")
        assert "abc123" in image_path and image_path.endswith(".png")
        assert "abc123" in flip_path and flip_path.endswith(".mp4")

    def test_windows_style_path(self):
        # Windows 反斜杠路径也要能正确提取文件名（旧版 split('/') 的 bug 已修）
        video_path, _, _ = file_name_extract(r"uploads\videos\abc123.mp4")
        assert "abc123" in video_path


class TestSafeResolve:
    def test_normal_filename(self):
        p = safe_resolve(settings.GENERATE_IMAGES_DIR, "abc.png")
        assert p.name == "abc.png"

    def test_path_traversal_blocked(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            safe_resolve(settings.GENERATE_IMAGES_DIR, "../../.env")

    def test_encoded_string_stays_inside(self):
        # %2F 在框架层会先被解码成 / 再到这里（已解码的形式由上一个用例覆盖）；
        # 到了本函数时 %2F 只是普通字符，文件仍被限制在目录内
        p = safe_resolve(settings.GENERATE_IMAGES_DIR, "..%2F..%2F.env")
        assert settings.GENERATE_IMAGES_DIR.resolve() in p.parents


class TestEditDistance:
    def test_identical_sentences(self):
        score, dist = pinyin_similarity_score("打开客厅灯", "打开客厅灯")
        assert score == 1.0
        assert dist == 0

    def test_similar_sentences_high_score(self):
        score, _ = pinyin_similarity_score("打开客厅灯", "打开卧室的灯")
        assert 0.5 <= score < 1.0

    def test_different_sentences_low_score(self):
        score, dist = pinyin_similarity_score("今天天气很好", "帮我把空调关掉")
        assert score < 0.5
        assert dist > 0

    def test_front_back_nasal_confusion(self):
        # “陈/程” 这类前后鼻音差异应被归一化，相似度要高
        score, _ = pinyin_similarity_score("我姓陈", "我姓程")
        assert score > 0.9

    def test_normalize(self):
        assert normalize_pinyin("lin") == "ning"   # in→ing 后 l→n，lin/ling/ning 归一到 ning
        assert normalize_pinyin("lan") == "nang"

    def test_empty(self):
        score, dist = pinyin_similarity_score("", "")
        assert score == 1.0 and dist == 0


class TestSettings:
    def test_common_video_exts(self):
        assert ".mp4" in settings.VIDEO_EXTS
        assert ".mov" in settings.VIDEO_EXTS

    def test_paths_under_base_dir(self):
        assert str(settings.VIDEO_DIR).startswith(str(settings.BASE_DIR))