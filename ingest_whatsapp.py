#!/usr/bin/env python3
"""
Paso 1 del pipeline RAG: parsear y limpiar un export .txt de WhatsApp (Android, ES).

Uso:
  python scripts/ingest_whatsapp.py "Chat de WhatsApp con Clau....txt"
  python scripts/ingest_whatsapp.py chat.txt -o output/clean.jsonl --stats output/stats.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

DEFAULT_MY_NAME = "Andres"
DEFAULT_PARTNER_RAW = "Clau Bombón De Licor🧸"
DEFAULT_PARTNER_DISPLAY = "Claudia"

# Línea de mensaje: 4/3/26, 20:52 - Andres: texto...
MSG_LINE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?)\s*"
    r"-\s+"
    r"([^:]+):\s*"
    r"(.*)$",
    re.IGNORECASE,
)

# Quitar marcas invisibles de WhatsApp (LRM, RLM, etc.)
INVISIBLE = re.compile(r"[\u200e\u200f\u202a-\u202e\ufeff]")

# Adjuntos con multimedia incluida en el export
ATTACHMENT = re.compile(
    r"^(?:"
    r"(?:IMG|VID|PTT|STK|AUD)-[\w-]+\.(?:jpg|jpeg|png|webp|mp4|opus|gif)"
    r"|.+?\.(?:pdf|docx?|xlsx?|pptx?|zip|rar)"
    r")\s*\(archivo adjunto\)\s*$",
    re.IGNORECASE,
)

MEDIA_PREFIX = re.compile(
    r"^(?:IMG|VID|PTT|STK|AUD)-[\w-]+\.(?:jpg|jpeg|png|webp|mp4|opus|gif)",
    re.IGNORECASE,
)

SYSTEM_PATTERNS = (
    re.compile(r"los mensajes y las llamadas están cifrados", re.I),
    re.compile(r"messages and calls are end-to-end encrypted", re.I),
    re.compile(r"creaste el grupo", re.I),
    re.compile(r"you created group", re.I),
    re.compile(r"cambió el icono del grupo", re.I),
    re.compile(r"cambió su número de teléfono", re.I),
)

DROP_BODY_PATTERNS = (
    re.compile(r"^se eliminó este mensaje\.?$", re.I),
    re.compile(r"^this message was deleted\.?$", re.I),
    re.compile(r"^ubicación en tiempo real compartida$", re.I),
    re.compile(r"^live location shared$", re.I),
    re.compile(r"^ubicación:\s*https?://", re.I),
    re.compile(r"^location:\s*https?://", re.I),
)

# Toda multimedia sin texto útil se descarta; solo entra texto real al RAG.
DROP_MEDIA_TYPES = frozenset(
    {"sticker", "audio", "image", "video", "unknown", "location"}
)


@dataclass
class Message:
    date: str
    time: str
    speaker: str
    text: str
    line_no: int


@dataclass
class CleanMessage:
    date: str
    time: str
    speaker: str
    text: str
    is_partner: bool | None = None


def normalize_text(value: str) -> str:
    text = INVISIBLE.sub("", value or "")
    return unicodedata.normalize("NFC", text).strip()


def classify_attachment(body: str) -> str:
    clean = normalize_text(body)
    upper = clean.upper()
    if "ARCHIVO ADJUNTO" not in clean.upper() and not ATTACHMENT.match(clean):
        return "text"
    if upper.startswith("PTT-") or ".OPUS" in upper:
        return "audio"
    if upper.startswith("IMG-"):
        return "image"
    if upper.startswith("VID-"):
        return "video"
    if upper.startswith("STK-"):
        return "sticker"
    if upper.startswith("AUD-"):
        return "audio"
    if ".PDF" in upper or ".DOC" in upper or ".XLS" in upper:
        return "document"
    if ATTACHMENT.match(clean):
        if MEDIA_PREFIX.match(clean.split("(", 1)[0].strip()):
            return "unknown"
        return "document"
    return "text"


def attachment_placeholder(kind: str, filename: str = "") -> str | None:
    if kind in DROP_MEDIA_TYPES:
        return None
    if kind == "document" and filename:
        name = filename.replace("(archivo adjunto)", "").strip()
        return name  # p. ej. guia_verbos.pdf — es texto informativo
    return None


def is_system_message(speaker: str, body: str) -> bool:
    blob = f"{speaker} {body}".lower()
    return any(p.search(blob) for p in SYSTEM_PATTERNS)


def should_drop_body(body: str) -> bool:
    return any(p.match(normalize_text(body)) for p in DROP_BODY_PATTERNS)


def parse_location_body(body: str) -> str | None:
    # Ubicaciones sin texto conversacional → descartar
    return None


def parse_export(lines: list[str]) -> list[Message]:
    messages: list[Message] = []
    current: Message | None = None

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n\r")
        match = MSG_LINE.match(line)
        if match:
            if current:
                messages.append(current)
            current = Message(
                date=match.group(1),
                time=match.group(2),
                speaker=normalize_text(match.group(3)),
                text=normalize_text(match.group(4)),
                line_no=i,
            )
            continue

        if current and line.strip():
            extra = normalize_text(line)
            current.text = f"{current.text}\n{extra}".strip() if current.text else extra

    if current:
        messages.append(current)
    return messages


def split_media_caption(text: str, media_kind: str) -> tuple[str | None, str]:
    """
    En Android, a veces la 'caption' de una imagen va en la línea siguiente
    sin cabecera de fecha. Ya viene fusionada en text tras parse_export.
    """
    if media_kind not in {"image", "video"}:
        return None, text

    lines = text.split("\n", 1)
    head = lines[0].strip()
    if not ATTACHMENT.match(head):
        return None, text

    if len(lines) == 1:
        return head, ""

    tail = lines[1].strip()
    # Evitar confundir con otra línea de adjunto suelta
    if ATTACHMENT.match(tail) or MEDIA_PREFIX.match(tail):
        return head, tail
    return head, tail


def normalize_speaker_name(speaker: str, raw_name: str, display_name: str) -> str:
    if speaker.casefold() == raw_name.casefold():
        return display_name
    return speaker


def replace_name_in_text(text: str, raw_name: str, display_name: str) -> str:
    if raw_name and raw_name in text:
        return text.replace(raw_name, display_name)
    return text


def clean_messages(
    messages: list[Message],
    *,
    my_name: str = DEFAULT_MY_NAME,
    partner_raw_name: str = DEFAULT_PARTNER_RAW,
    partner_display_name: str = DEFAULT_PARTNER_DISPLAY,
) -> tuple[list[CleanMessage], dict]:
    stats = {
        "input_messages": len(messages),
        "removed_system": 0,
        "removed_deleted": 0,
        "removed_media": 0,
        "kept_captions": 0,
        "kept_text": 0,
        "output_messages": 0,
        "by_speaker": {},
    }

    my_key = my_name.casefold()
    partner_key = partner_raw_name.casefold()
    cleaned: list[CleanMessage] = []

    for msg in messages:
        speaker = msg.speaker
        body = msg.text

        if is_system_message(speaker, body):
            stats["removed_system"] += 1
            continue

        if should_drop_body(body):
            stats["removed_deleted"] += 1
            continue

        loc = parse_location_body(body)
        if loc:
            body = loc
            media_kind = "location"
        else:
            media_kind = classify_attachment(body.split("\n", 1)[0])

        if media_kind == "text":
            stats["kept_text"] += 1
            final_text = body
        else:
            media_line, caption = split_media_caption(body, media_kind)
            if ATTACHMENT.match(normalize_text(body.split("\n", 1)[0])):
                lines = body.split("\n", 1)
                if len(lines) == 1 or ATTACHMENT.match(normalize_text(lines[1])):
                    caption = ""

            placeholder = attachment_placeholder(media_kind, media_line or body)

            if caption:
                # Caption bajo imagen/vídeo: solo el texto, sin marcador de media
                final_text = caption
                stats["kept_captions"] += 1
            elif placeholder:
                final_text = placeholder
            else:
                stats["removed_media"] += 1
                continue

        if not final_text.strip():
            stats["removed_media"] += 1
            continue

        final_text = replace_name_in_text(
            final_text, partner_raw_name, partner_display_name
        )

        is_partner = None
        sk = msg.speaker.casefold()
        if sk == partner_key:
            is_partner = True
        elif sk == my_key:
            is_partner = False

        speaker = normalize_speaker_name(
            speaker, partner_raw_name, partner_display_name
        )

        cleaned.append(
            CleanMessage(
                date=msg.date,
                time=msg.time,
                speaker=speaker,
                text=final_text,
                is_partner=is_partner,
            )
        )
        stats["by_speaker"][speaker] = stats["by_speaker"].get(speaker, 0) + 1

    stats["output_messages"] = len(cleaned)
    return cleaned, stats


def iter_jsonl(messages: list[CleanMessage]) -> Iterator[str]:
    for msg in messages:
        yield json.dumps(asdict(msg), ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpiar export WhatsApp .txt")
    parser.add_argument("input", type=Path, help="Archivo .txt exportado")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/whatsapp_clean.jsonl"),
        help="JSONL de mensajes limpios",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("output/whatsapp_stats.json"),
        help="Resumen de limpieza",
    )
    parser.add_argument("--my-name", default=DEFAULT_MY_NAME)
    parser.add_argument(
        "--partner-raw-name",
        default=DEFAULT_PARTNER_RAW,
        help="Nombre tal como aparece en el export de WhatsApp",
    )
    parser.add_argument(
        "--partner-display-name",
        default=DEFAULT_PARTNER_DISPLAY,
        help="Nombre normalizado en el RAG (p. ej. Claudia)",
    )
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    parsed = parse_export(lines)
    cleaned, stats = clean_messages(
        parsed,
        my_name=args.my_name,
        partner_raw_name=args.partner_raw_name,
        partner_display_name=args.partner_display_name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(iter_jsonl(cleaned)) + "\n", encoding="utf-8")
    args.stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Entrada:  {stats['input_messages']} mensajes parseados")
    print(f"Salida:   {stats['output_messages']} mensajes de texto")
    print(f"Quitados: {stats['removed_media']} multimedia, {stats['removed_deleted']} borrados, {stats['removed_system']} sistema")
    print(f"Captions: {stats['kept_captions']} textos bajo imágenes/vídeos conservados")
    print(f"Speaker alias applied -> {args.partner_display_name!r}")
    print(f"Guardado: {args.output}")
    print(f"Stats:    {args.stats}")


if __name__ == "__main__":
    main()
