"""Tag photos via the fleet's vision model.

Walks a directory of images, calls the LLM router's vision-capable model
once per photo, and appends a JSON record to a JSONL file. Resumable:
files whose sha256 already appears in the output are skipped.

CLI:
    uv run python -m llm_router.photo_tag <dir> [--out tags.jsonl] [...]

Routing: defaults to the LiteLLM proxy at euclid.local:4010, model
`vision-fast` (Qwen3-VL-8B on hypatia after the 2026-06-03 deploy).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

try:
    import pillow_heif  # type: ignore[import-untyped]

    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


class PhotoTags(BaseModel):
    hair_color: str = Field(
        description="One of: blonde, brown, black, red, gray, white, bald, other, not_visible"
    )
    hair_length: str = Field(
        description="One of: short, medium, long, shaved, not_visible"
    )
    eye_color: str = Field(
        description="One of: blue, brown, green, hazel, gray, other, not_visible"
    )
    skin_tone: str = Field(
        description="One of: fair, medium, olive, tan, dark, other, not_visible"
    )
    setting: str = Field(
        description="One of: indoor, outdoor, mixed, unclear"
    )
    location_type: str = Field(
        description="Short noun phrase, e.g. beach, park, home, office, restaurant, studio, vehicle, sports_venue, street, unclear"
    )
    clothing_style: str = Field(
        description="Short adjective, e.g. casual, formal, business, athletic, swimwear, uniform, costume, unclear"
    )
    notes: str = Field(
        default="",
        description="Optional free-text observation (<=120 chars); empty string if none.",
    )


SYSTEM_PROMPT = """\
You are a precise photo-tagging assistant. Given a single photo, describe the
SUBJECT (the person closest to the camera, or the most prominent person if
multiple) and the SCENE. Return a single JSON object with the requested fields
and nothing else.

Rules:
- Use the listed value for each field when one applies. Otherwise use a short
  lowercase word with underscores instead of spaces.
- For physical traits that cannot be determined from the image, return
  "not_visible". For setting that cannot be determined, return "unclear".
- Never invent details. Never describe features that aren't visible.
- Do not include identifying information (names, identities) even if you
  recognize someone.
"""


FIELD_GUIDE = """\
Return a single JSON object with EXACTLY these top-level keys and nothing else:
- hair_color: one of [blonde, brown, black, red, gray, white, bald, other, not_visible]
- hair_length: one of [short, medium, long, shaved, not_visible]
- eye_color: one of [blue, brown, green, hazel, gray, other, not_visible]
- skin_tone: one of [fair, medium, olive, tan, dark, other, not_visible]
- setting: one of [indoor, outdoor, mixed, unclear]
- location_type: short snake_case noun, e.g. beach, park, home, office, restaurant, studio, vehicle, street, sports_venue, unclear
- clothing_style: short snake_case adjective, e.g. casual, formal, business, athletic, swimwear, uniform, costume, unclear
- notes: short observation under 120 chars, or empty string ""

Example output (the entire response must look like this, no schema wrapper, no markdown, no commentary):
{"hair_color":"brown","hair_length":"medium","eye_color":"blue","skin_tone":"fair","setting":"outdoor","location_type":"park","clothing_style":"casual","notes":""}
"""


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def encode_image(path: Path, max_side: int = 1024) -> str:
    img = Image.open(path)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_existing_hashes(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen: set[str] = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = rec.get("sha256")
            if isinstance(sha, str):
                seen.add(sha)
    return seen


REQUIRED_KEYS = {
    "hair_color",
    "hair_length",
    "eye_color",
    "skin_tone",
    "setting",
    "location_type",
    "clothing_style",
}


def _unwrap_schema_echo(data: Any) -> Any:
    """Some small models echo the JSON Schema and put values under "properties".

    If the parsed payload looks schema-shaped (top-level has "properties" and
    no required keys, but properties does have them), return the inner dict.
    """
    if (
        isinstance(data, dict)
        and isinstance(data.get("properties"), dict)
        and not (REQUIRED_KEYS & data.keys())
        and (REQUIRED_KEYS & data["properties"].keys())
    ):
        return data["properties"]
    return data


async def tag_one(
    client: AsyncOpenAI, path: Path, model: str, max_side: int
) -> PhotoTags:
    data_url = encode_image(path, max_side=max_side)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FIELD_GUIDE},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=300,
    )
    raw = response.choices[0].message.content or ""
    data = _unwrap_schema_echo(json.loads(raw))
    return PhotoTags.model_validate(data)


async def main_async(args: argparse.Namespace) -> int:
    photo_dir = Path(args.dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not photo_dir.is_dir():
        print(f"Not a directory: {photo_dir}", file=sys.stderr)
        return 2

    photos = sorted(
        p
        for p in photo_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not photos:
        print(f"No images in {photo_dir} (looked for {sorted(IMAGE_EXTS)})", file=sys.stderr)
        return 1

    seen = load_existing_hashes(out_path)
    print(
        f"{len(photos)} images in {photo_dir}; {len(seen)} already tagged in {out_path.name}.",
        file=sys.stderr,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    done = 0
    failed = 0
    skipped = 0
    total = len(photos)

    async def process(path: Path) -> None:
        nonlocal done, failed, skipped
        sha = hash_file(path)
        if sha in seen:
            skipped += 1
            return
        async with sem:
            try:
                tags = await tag_one(client, path, args.model, args.max_side)
            except Exception as e:
                failed += 1
                print(f"FAIL {path.name}: {type(e).__name__}: {e}", file=sys.stderr)
                return
        rec: dict[str, Any] = {
            "filename": str(path.relative_to(photo_dir)),
            "sha256": sha,
            "size_bytes": path.stat().st_size,
            "tags": tags.model_dump(),
            "model": args.model,
            "tagged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line)
            seen.add(sha)
            done += 1
            print(
                f"[{done + skipped + failed}/{total}] {path.name}: "
                f"{tags.hair_color}/{tags.eye_color}/{tags.setting}/{tags.location_type}",
                file=sys.stderr,
            )

    await asyncio.gather(*(process(p) for p in photos))

    print(
        f"\nDone. tagged={done} skipped={skipped} failed={failed} total={total}",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 3


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="photo-tag",
        description="Tag photos via the fleet vision model (Qwen3-VL by default).",
    )
    ap.add_argument("dir", help="Directory of photos to tag (recursive).")
    ap.add_argument(
        "--out", default="tags.jsonl", help="Output JSONL path (default: tags.jsonl)."
    )
    ap.add_argument(
        "--model",
        default="vision-fast",
        help="LLM router model alias (default: vision-fast = Qwen3-VL-8B).",
    )
    ap.add_argument(
        "--base-url",
        default="http://euclid.local:4010/v1",
        help="LiteLLM proxy base URL.",
    )
    ap.add_argument(
        "--api-key",
        default="sk-litellm-master",
        help="LiteLLM master key.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent in-flight tag requests (default: 4).",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Resize images so longest side <= this many px before upload.",
    )
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
