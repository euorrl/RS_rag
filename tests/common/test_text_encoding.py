from pathlib import Path

import pytest

pytestmark = pytest.mark.common

TEXT_EXTENSIONS = {
    ".md",
    ".py",
    ".toml",
    ".ini",
    ".yml",
    ".yaml",
    ".txt",
}

TEXT_ROOTS = [
    Path("app"),
    Path("tests"),
    Path("docs"),
]

TEXT_FILES = [
    Path("README.md"),
    Path("pyproject.toml"),
    Path("pytest.ini"),
    Path("requirements.txt"),
]

MOJIBAKE_MARKERS = [
    "杩欐槸",
    "楠岃瘉",
    "妯″潡",
    "銆?",
    "鐨",
]


def iter_project_text_files():
    """遍历项目中需要按 UTF-8 维护的文本文件。"""
    current_file = Path(__file__).resolve().relative_to(Path.cwd())

    for root in TEXT_ROOTS:
        yield from (
            path
            for path in root.rglob("*")
            if (
                path.is_file()
                and path.suffix in TEXT_EXTENSIONS
                and path != current_file
            )
        )

    yield from TEXT_FILES


def test_project_text_files_are_utf8_without_mojibake():
    """验证项目文本文件均可按 UTF-8 读取，且不包含常见乱码片段。"""
    for path in iter_project_text_files():
        text = path.read_text(encoding="utf-8")

        assert "\ufffd" not in text, f"{path} contains replacement characters"

        for marker in MOJIBAKE_MARKERS:
            assert marker not in text, f"{path} contains mojibake marker {marker!r}"
