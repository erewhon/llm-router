"""AST-aware code chunking via tree-sitter, with whole-file fallback."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import tree_sitter_bash
import tree_sitter_python
from pydantic import BaseModel
from tree_sitter import Language, Node, Parser

PY_LANG = Language(tree_sitter_python.language())
BASH_LANG = Language(tree_sitter_bash.language())
PY_PARSER = Parser(PY_LANG)
BASH_PARSER = Parser(BASH_LANG)

EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".sh": "bash",
    ".bash": "bash",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".service": "ini",
    ".rs": "rust",
    ".go": "go",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
}
FILENAME_LANG: dict[str, str] = {
    "justfile": "just",
    "Justfile": "just",
    "Dockerfile": "docker",
    "Makefile": "make",
}

PY_CHUNK_TYPES = {"function_definition", "class_definition", "decorated_definition"}
BASH_CHUNK_TYPES = {"function_definition"}

MAX_FILE_BYTES = 200_000
WHOLE_FILE_MAX_LINES = 400
SLICE_LINES = 60
SLICE_OVERLAP = 10


class Chunk(BaseModel):
    repo: str
    path: str
    lang: str
    chunk_kind: str  # function | class | whole | slice
    name: str
    start_line: int
    end_line: int
    text: str
    mtime: float


def detect_lang(path: Path) -> str | None:
    return EXT_LANG.get(path.suffix.lower()) or FILENAME_LANG.get(path.name)


def _node_name(node: Node) -> str:
    inner = node
    if node.type == "decorated_definition":
        for c in node.children:
            if c.type in {"function_definition", "class_definition"}:
                inner = c
                break
    for c in inner.children:
        if c.type == "identifier":
            return c.text.decode("utf-8", errors="replace")
    return ""


def _python_kind(node: Node) -> str:
    if node.type == "class_definition":
        return "class"
    if node.type == "function_definition":
        return "function"
    for c in node.children:
        if c.type == "class_definition":
            return "class"
    return "function"


def _python_chunks(source: bytes, repo: str, path: str, mtime: float) -> Iterator[Chunk]:
    tree = PY_PARSER.parse(source)
    for node in tree.root_node.children:
        if node.type in PY_CHUNK_TYPES:
            text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            yield Chunk(
                repo=repo,
                path=path,
                lang="python",
                chunk_kind=_python_kind(node),
                name=_node_name(node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                text=text,
                mtime=mtime,
            )


def _bash_chunks(source: bytes, repo: str, path: str, mtime: float) -> Iterator[Chunk]:
    tree = BASH_PARSER.parse(source)
    for node in tree.root_node.children:
        if node.type in BASH_CHUNK_TYPES:
            text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            yield Chunk(
                repo=repo,
                path=path,
                lang="bash",
                chunk_kind="function",
                name=_node_name(node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                text=text,
                mtime=mtime,
            )


def _whole_or_sliced(
    source: bytes, lang: str, repo: str, path: str, mtime: float
) -> Iterator[Chunk]:
    text = source.decode("utf-8", errors="replace")
    if not text.strip():
        return
    lines = text.splitlines()
    n = len(lines)
    if n <= WHOLE_FILE_MAX_LINES:
        yield Chunk(
            repo=repo,
            path=path,
            lang=lang,
            chunk_kind="whole",
            name="",
            start_line=1,
            end_line=max(n, 1),
            text=text,
            mtime=mtime,
        )
        return
    step = SLICE_LINES - SLICE_OVERLAP
    for start in range(0, n, step):
        end = min(start + SLICE_LINES, n)
        slice_text = "\n".join(lines[start:end])
        if not slice_text.strip():
            continue
        yield Chunk(
            repo=repo,
            path=path,
            lang=lang,
            chunk_kind="slice",
            name="",
            start_line=start + 1,
            end_line=end,
            text=slice_text,
            mtime=mtime,
        )
        if end >= n:
            break


def chunk_file(repo: str, repo_root: Path, abs_path: Path) -> list[Chunk]:
    try:
        data = abs_path.read_bytes()
    except OSError:
        return []
    if not data or len(data) > MAX_FILE_BYTES:
        return []
    if b"\x00" in data[:8192]:
        return []
    rel = abs_path.relative_to(repo_root).as_posix()
    mtime = abs_path.stat().st_mtime
    lang = detect_lang(abs_path) or "text"

    if lang == "python":
        chunks = list(_python_chunks(data, repo, rel, mtime))
        if chunks:
            return chunks
        return list(_whole_or_sliced(data, "python", repo, rel, mtime))
    if lang == "bash":
        fn_chunks = list(_bash_chunks(data, repo, rel, mtime))
        whole = list(_whole_or_sliced(data, "bash", repo, rel, mtime))
        return fn_chunks + whole
    return list(_whole_or_sliced(data, lang, repo, rel, mtime))
