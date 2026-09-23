from __future__ import annotations

import pytest

from app.helpers import tokenize

pytestmark = pytest.mark.unit


def test_tokenize_lowercases_words() -> None:
    assert tokenize("Hello, Zephyr-1!") == ["hello", "zephyr", "1"]
