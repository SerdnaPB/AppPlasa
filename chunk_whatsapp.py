#!/usr/bin/env python3
"""
Paso 2 del pipeline RAG: trocear mensajes limpios en chunks para indexar.

Política por defecto (opción A):
  - Máx. 1 000 caracteres o 25 mensajes por chunk (lo que ocurra antes)
  - Solapamiento de 3 mensajes entre chunks consecutivos
  - Nuevo chunk si hay más de 4 h entre mensajes

Uso:
  python scripts/chunk_whatsapp.py
  python scripts/chunk_whatsapp.py output/whatsapp_clean.jsonl -o output/whatsapp_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_MAX_CHARS = 1000
DEFAULT_MAX_MESSAGES = 25
DEFAULT_OVERLAP = 3
DEFAULT_GAP_HOURS = 4

TIME_RE = re.compile(
    r"^(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([ap])\.?m\.?)?$",
    re.IGNORECASE,
)


@dataclass
class CleanMessage:
    date: str
    time: str
    speaker: str
    text: str
    is_partner: bool | None = None


@dataclass
class Chunk:
    id: str
    content: str
    message_count: int
    char_count: int
    date_start: str
    time_start: str
    date_end: str
    time_end: str
    partner_messages: int
    start_index: int
    end_index: int


def load_messages(path: Path) -> list[CleanMessage]:
    messages: list[CleanMessage] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages.append(CleanMessage(**row))
    return messages


def parse_datetime(date_str: str, time_str: str) -> datetime | None:
    try:
        day, month, year = (int(x) for x in date_str.split("/"))
        if year < 100:
            year += 2000
        match = TIME_RE.match(time_str.strip())
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3) or 0)
        ampm = (match.group(4) or "").lower()
        if ampm == "p" and hour < 12:
            hour += 12
        if ampm == "a" and hour == 12:
            hour = 0
        return datetime(year, month, day, hour, minute, second)
    except (ValueError, TypeError):
        return None


def format_line(msg: CleanMessage) -> str:
    return f"[{msg.date} {msg.time}] {msg.speaker}: {msg.text}"


def build_chunks(
    messages: list[CleanMessage],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    overlap: int = DEFAULT_OVERLAP,
    gap_hours: float = DEFAULT_GAP_HOURS,
) -> list[Chunk]:
    if not messages:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_num = 0
    gap_delta = timedelta(hours=gap_hours)

    while start < len(messages):
        end = start
        selected: list[CleanMessage] = []
        char_count = 0

        while end < len(messages):
            if selected:
                prev_dt = parse_datetime(messages[end - 1].date, messages[end - 1].time)
                curr_dt = parse_datetime(messages[end].date, messages[end].time)
                if prev_dt and curr_dt and (curr_dt - prev_dt) > gap_delta:
                    break

            line = format_line(messages[end])
            line_len = len(line) + 1

            if selected and (
                char_count + line_len > max_chars or len(selected) >= max_messages
            ):
                break

            selected.append(messages[end])
            char_count += line_len
            end += 1

        if not selected:
            start += 1
            continue

        partner_count = sum(1 for m in selected if m.is_partner is True)
        content = "\n".join(format_line(m) for m in selected)
        chunk_num += 1
        chunks.append(
            Chunk(
                id=f"chunk-{chunk_num:05d}",
                content=content,
                message_count=len(selected),
                char_count=len(content),
                date_start=selected[0].date,
                time_start=selected[0].time,
                date_end=selected[-1].date,
                time_end=selected[-1].time,
                partner_messages=partner_count,
                start_index=start,
                end_index=end,
            )
        )

        if end >= len(messages):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def summarize(chunks: list[Chunk]) -> dict[str, Any]:
    if not chunks:
        return {"chunk_count": 0}

    sizes = [c.char_count for c in chunks]
    counts = [c.message_count for c in chunks]
    sizes.sort()
    counts.sort()
    n = len(chunks)

    return {
        "chunk_count": n,
        "total_chars": sum(sizes),
        "messages_covered": sum(counts),
        "chars_min": sizes[0],
        "chars_p50": sizes[n // 2],
        "chars_p95": sizes[int(n * 0.95)],
        "chars_max": sizes[-1],
        "msgs_min": counts[0],
        "msgs_p50": counts[n // 2],
        "msgs_p95": counts[int(n * 0.95)],
        "msgs_max": counts[-1],
        "avg_chars": round(sum(sizes) / n, 1),
        "avg_messages": round(sum(counts) / n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trocear mensajes limpios en chunks RAG")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("output/whatsapp_clean.jsonl"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/whatsapp_chunks.jsonl"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("output/whatsapp_chunks_stats.json"),
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--gap-hours", type=float, default=DEFAULT_GAP_HOURS)
    args = parser.parse_args()

    messages = load_messages(args.input)
    chunks = build_chunks(
        messages,
        max_chars=args.max_chars,
        max_messages=args.max_messages,
        overlap=args.overlap,
        gap_hours=args.gap_hours,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    stats = summarize(chunks)
    stats["settings"] = {
        "max_chars": args.max_chars,
        "max_messages": args.max_messages,
        "overlap": args.overlap,
        "gap_hours": args.gap_hours,
        "input_messages": len(messages),
    }
    args.stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Mensajes:  {len(messages)}")
    print(f"Chunks:    {stats['chunk_count']}")
    print(f"Chars:     p50={stats['chars_p50']} p95={stats['chars_p95']} max={stats['chars_max']}")
    print(f"Mensajes/chunk: p50={stats['msgs_p50']} max={stats['msgs_max']}")
    print(f"Guardado:  {args.output}")
    print(f"Stats:     {args.stats}")


if __name__ == "__main__":
    main()
