from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import ENV_FILE_PATH, PROJECT_ROOT, Settings

pytestmark = pytest.mark.unit

_REQUIRED_ENV_KEYS = (
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
)


def _clear_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _REQUIRED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _write_env(path: Path, *, embedding_model: str = "temp-embedder") -> Path:
    path.write_text(
        "\n".join(
            [
                "QDRANT_URL=http://localhost:6333",
                "QDRANT_COLLECTION=rag_chunks",
                f"EMBEDDING_MODEL={embedding_model}",
                "RERANKER_MODEL=temp-reranker",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_settings_env_file_is_project_root_absolute() -> None:
    assert ENV_FILE_PATH == PROJECT_ROOT / ".env"
    assert ENV_FILE_PATH.is_absolute()
    assert (PROJECT_ROOT / "app").is_dir()
    assert (PROJECT_ROOT / ".env.example").is_file()


def test_settings_loads_injected_env_file_when_cwd_has_no_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = _write_env(tmp_path / "injected.env", embedding_model="from-temp-env")
    cwd = tmp_path / "no_env_here"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    _clear_required_env(monkeypatch)

    settings = Settings(_env_file=env_file)

    assert settings.embedding_model == "from-temp-env"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "rag_chunks"
    assert settings.reranker_model == "temp-reranker"


def test_missing_embedding_model_raises_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_required_env(monkeypatch)

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            qdrant_url="http://localhost:6333",
            qdrant_collection="rag_chunks",
            reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        )

    message = str(exc_info.value)
    assert "embedding_model" in message or "EMBEDDING_MODEL" in message


def test_empty_embedding_model_raises_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_required_env(monkeypatch)

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            qdrant_url="http://localhost:6333",
            qdrant_collection="rag_chunks",
            embedding_model="   ",
            reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        )

    message = str(exc_info.value)
    assert "embedding_model" in message or "EMBEDDING_MODEL" in message


def test_relative_bm25_path_resolves_against_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_required_env(monkeypatch)

    settings = Settings(
        _env_file=None,
        qdrant_url="http://localhost:6333",
        qdrant_collection="rag_chunks",
        embedding_model="fake",
        reranker_model="fake-reranker",
        bm25_index_path=Path("data/bm25_index.json"),
    )

    expected = (PROJECT_ROOT / "data" / "bm25_index.json").resolve()
    assert settings.bm25_index_path == expected
    assert settings.bm25_index_path.is_absolute()
    assert tmp_path.resolve() not in settings.bm25_index_path.parents
