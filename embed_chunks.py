#!/usr/bin/env python3
"""
Paso 3 del pipeline RAG: embeddings gratuitos en local con Hugging Face
(sentence-transformers) y subida opcional a Supabase pgvector.

Modelo por defecto: paraphrase-multilingual-MiniLM-L12-v2
  - Gratis, multilingüe (español), 384 dimensiones
  - Primera ejecución: descarga el modelo (~120 MB)

Uso:
  pip install -r requirements-rag.txt
  cp .env.example .env   # y rellena SUPABASE_SERVICE_KEY

  python scripts/embed_chunks.py
  python scripts/embed_chunks.py --skip-upload          # solo embeddings locales
  python scripts/embed_chunks.py --upload-only          # subir JSONL ya generado
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INPUT = Path("output/whatsapp_chunks.jsonl")
DEFAULT_OUTPUT = Path("output/whatsapp_embedded.jsonl")
DEFAULT_BATCH = 64
DEFAULT_UPLOAD_BATCH = 100


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_embedded_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def append_embedded(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_batches(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    print(f"Cargando modelo: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"Dimensiones: {dim}")

    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        end = min(start + batch_size, total)
        print(f"  Embedding {start + 1}-{end}/{total}…")
        encoded = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors.extend(encoded.tolist())

    return vectors


def build_embedded_rows(
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    short_name = model_name.split("/")[-1]
    for chunk, vector in zip(chunks, vectors):
        rows.append(
            {
                "id": chunk["id"],
                "content": chunk["content"],
                "embedding": vector,
                "embedding_model": short_name,
                "message_count": chunk.get("message_count"),
                "char_count": chunk.get("char_count"),
                "date_start": chunk.get("date_start"),
                "time_start": chunk.get("time_start"),
                "date_end": chunk.get("date_end"),
                "time_end": chunk.get("time_end"),
                "partner_messages": chunk.get("partner_messages"),
                "start_index": chunk.get("start_index"),
                "end_index": chunk.get("end_index"),
            }
        )
    return rows


def validate_supabase_key(key: str) -> None:
    placeholders = {"", "your_service_role_key_here", "peg_aqui_la_service_role_key_completa"}
    if key.strip() in placeholders:
        raise SystemExit(
            "SUPABASE_SERVICE_KEY en .env sigue siendo el placeholder.\n"
            "Copia la key service_role desde Supabase → Settings → API y guarda .env."
        )
    if len(key) < 100:
        raise SystemExit(
            f"SUPABASE_SERVICE_KEY parece incompleta (longitud {len(key)}).\n"
            "La JWT real tiene ~200+ caracteres. Ponla entre comillas en .env:\n"
            '  SUPABASE_SERVICE_KEY="eyJhbGci..."'
        )
    if "service_role" not in key and "." in key:
        import base64
        try:
            payload = key.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload).decode("utf-8", errors="ignore")
            if "service_role" not in decoded:
                raise SystemExit(
                    "Parece que pegaste la key anon en lugar de service_role.\n"
                    "En Supabase → Settings → API usa la fila service_role (secret)."
                )
        except Exception:
            pass


def upload_to_supabase(rows: list[dict[str, Any]], batch_size: int) -> None:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        raise SystemExit(
            "Faltan SUPABASE_URL y SUPABASE_SERVICE_KEY en .env para subir a Supabase."
        )
    validate_supabase_key(key)

    client = create_client(url, key)
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        end = min(start + batch_size, total)
        print(f"  Subiendo {start + 1}-{end}/{total}…")
        try:
            client.table("chat_chunks").upsert(batch, on_conflict="id").execute()
        except Exception as err:
            msg = str(err).lower()
            if "invalid api key" in msg or "401" in msg:
                raise SystemExit(
                    "Supabase rechazó la API key (401).\n"
                    "Revisa .env: service_role completa, entre comillas, y archivo guardado."
                ) from err
            if "chat_chunks" in msg and ("pgrst205" in msg or "could not find the table" in msg):
                raise SystemExit(
                    "No existe la tabla public.chat_chunks.\n"
                    "Ejecuta supabase/chat_chunks.sql en Supabase → SQL Editor → Run."
                ) from err
            raise
        time.sleep(0.15)


def load_all_embedded(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Embeddings gratuitos (Hugging Face local) + Supabase"
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--upload-batch", type=int, default=DEFAULT_UPLOAD_BATCH)
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Solo generar embeddings locales",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Subir output existente sin re-embeder",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embeder aunque ya exista salida (sobrescribe output)",
    )
    args = parser.parse_args()

    if args.upload_only:
        if not args.output.exists():
            raise SystemExit(f"No existe {args.output}. Ejecuta primero sin --upload-only.")
        rows = load_all_embedded(args.output)
        print(f"Subiendo {len(rows)} chunks a Supabase…")
        upload_to_supabase(rows, args.upload_batch)
        print("Subida completada.")
        return

    if not args.input.exists():
        raise SystemExit(f"No existe {args.input}. Ejecuta chunk_whatsapp.py antes.")

    chunks = load_chunks(args.input)
    done_ids = set() if args.force else load_embedded_ids(args.output)
    pending = [c for c in chunks if c["id"] not in done_ids]

    print(f"Chunks totales: {len(chunks)}")
    print(f"Ya embedidos:   {len(done_ids)}")
    print(f"Pendientes:     {len(pending)}")

    if pending:
        texts = [c["content"] for c in pending]
        vectors = encode_batches(texts, args.model, args.batch_size)
        rows = build_embedded_rows(pending, vectors, args.model)

        if args.force and args.output.exists():
            args.output.unlink()

        append_embedded(args.output, rows)
        print(f"Guardado en {args.output} (+{len(rows)} filas)")
    else:
        print("Nada pendiente de embeder.")

    if not args.skip_upload:
        all_rows = load_all_embedded(args.output)
        print(f"Subiendo {len(all_rows)} chunks a Supabase…")
        upload_to_supabase(all_rows, args.upload_batch)
        print("Subida completada.")
        print(
            "\nSiguiente: en Supabase SQL Editor, crea el índice ivfflat "
            "(ver comentario al final de supabase/chat_chunks.sql)."
        )
    else:
        print("Subida omitida (--skip-upload).")


if __name__ == "__main__":
    main()
