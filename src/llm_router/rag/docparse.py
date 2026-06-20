"""Document parsing + chunking via Docling, with a plain-text fallback.

Docling handles the rich formats (PDF/DOCX/PPTX/HTML/Markdown) — layout,
tables, headings — and exports a structured document we chunk with its
HybridChunker (token-aware, heading-contextualised). Plain `.txt`/`.rst`
files skip Docling and use a simple paragraph-packing chunker.

Docling and its models are heavy, so all docling imports are lazy: importing
this module is cheap; the converter/chunker are built on first use.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

# Rich formats routed through Docling; plain-text formats use the fallback.
DOCLING_EXTS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".markdown", ".adoc"}
TEXT_EXTS = {".txt", ".text", ".rst"}

# Plain-text fallback packing.
TEXT_CHUNK_CHARS = 1500
TEXT_OVERLAP_CHARS = 200

_converter = None
_chunker = None


class DocChunk(BaseModel):
    source: str  # path relative to the ingest root (posix)
    title: str  # heading breadcrumb or document name — context for retrieval
    topics: list[str]  # multi-valued topic tags (politics / astronomy / technology)
    chunk_index: int
    page: int | None
    text: str  # contextualised text, used for both embedding and display
    mtime: float


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in (DOCLING_EXTS | TEXT_EXTS)


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter

        _converter = DocumentConverter()
    return _converter


def _get_chunker():
    global _chunker
    if _chunker is None:
        try:
            from docling.chunking import HybridChunker

            _chunker = HybridChunker()
        except Exception:
            from docling.chunking import HierarchicalChunker

            _chunker = HierarchicalChunker()
    return _chunker


def _chunk_page(chunk) -> int | None:
    try:
        return chunk.meta.doc_items[0].prov[0].page_no
    except (AttributeError, IndexError, TypeError):
        return None


def _docling_chunks(
    abs_path: Path, source: str, mtime: float, topics: list[str]
) -> Iterator[DocChunk]:
    result = _get_converter().convert(str(abs_path))
    doc = result.document
    chunker = _get_chunker()
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        try:
            text = chunker.contextualize(chunk=ch)
        except Exception:
            text = ch.text
        if not text or not text.strip():
            continue
        headings = getattr(ch.meta, "headings", None) or []
        yield DocChunk(
            source=source,
            title=" > ".join(headings) if headings else abs_path.stem,
            topics=topics,
            chunk_index=i,
            page=_chunk_page(ch),
            text=text,
            mtime=mtime,
        )


def _text_chunks(
    abs_path: Path, source: str, mtime: float, topics: list[str]
) -> Iterator[DocChunk]:
    raw = abs_path.read_text(encoding="utf-8", errors="replace")
    paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
    packed: list[str] = []
    buf = ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > TEXT_CHUNK_CHARS:
            packed.append(buf)
            tail = buf[-TEXT_OVERLAP_CHARS:]
            buf = f"{tail}\n\n{p}"
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        packed.append(buf)
    for i, text in enumerate(packed):
        yield DocChunk(
            source=source,
            title=abs_path.stem,
            topics=topics,
            chunk_index=i,
            page=None,
            text=text,
            mtime=mtime,
        )


def chunk_document(root: Path, abs_path: Path, topics: list[str]) -> list[DocChunk]:
    """Parse + chunk one document. `root` anchors the relative `source` path."""
    try:
        source = abs_path.relative_to(root).as_posix()
    except ValueError:
        source = abs_path.name
    mtime = abs_path.stat().st_mtime
    ext = abs_path.suffix.lower()
    if ext in TEXT_EXTS:
        return list(_text_chunks(abs_path, source, mtime, topics))
    if ext in DOCLING_EXTS:
        return list(_docling_chunks(abs_path, source, mtime, topics))
    return []
