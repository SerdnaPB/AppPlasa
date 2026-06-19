#!/usr/bin/env python3
"""
Paso 4 del pipeline RAG: preguntar sobre Claudia usando chunks en Supabase + Gemini.

Flujo:
  1. Embedding local de la pregunta (mismo modelo HF que en embed_chunks.py)
  2. Búsqueda vectorial en Supabase (match_chat_chunks)
  3. Respuesta con Gemini usando solo los fragmentos recuperados

Uso:
  python scripts/ask_partner.py "¿Qué planes de viaje mencionó Claudia?"
  python scripts/ask_partner.py -i
  python scripts/ask_partner.py "..." --show-sources
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from dotenv import load_dotenv

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TOP_K = 12
DEFAULT_VECTOR_POOL = 20
DEFAULT_PARTNER = "Claudia"
DEFAULT_MY_NAME = "Andres"

STOPWORDS = frozenset(
    """
    quien quienes como cuando donde cual cuales que qué sobre con para por del
    de la las los una uno unos unas tiene tienen haber esta este estos esas
    claudia andres musica música gusta gustan cosas algo muy mas más
    quedado quedó quedo junio julio agosto septiembre octubre noviembre diciembre
    enero febrero marzo abril mayo menciona mencionar
    """.split()
)

MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def extract_date_patterns(question: str) -> list[str]:
    patterns: list[str] = []
    lower = question.casefold()

    for match in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", question):
        day, month, year = match.groups()
        if year:
            patterns.append(f"{int(day)}/{int(month)}/{year[-2:]}")
        patterns.append(f"{int(day)}/{int(month)}")

    for match in re.finditer(
        r"\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b",
        lower,
    ):
        day = int(match.group(1))
        month = MONTHS[match.group(2)]
        patterns.append(f"{day}/{month}")
        patterns.append(f"{day}/{month}/26")

    deduped: list[str] = []
    for item in patterns:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]


def keyword_variants(word: str) -> list[str]:
    variants = {word, word.casefold(), strip_accents(word).casefold()}
    return [v for v in variants if len(v) >= 3]


def extract_keywords(question: str, partner_name: str, my_name: str) -> list[str]:
    words = re.findall(r"[\wáéíóúñÁÉÍÓÚÑ]+", question, flags=re.IGNORECASE)
    skip = STOPWORDS | {partner_name.casefold(), my_name.casefold()}
    keywords: list[str] = []

    for word in words:
        key = word.casefold()
        if len(key) < 3 or key in skip:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= 6:
            break

    keywords.extend(extract_date_patterns(question))
    return keywords[:8]


def encode_query(text: str, model_name: str) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector.tolist()


def get_supabase_client():
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_ANON_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    )
    if not url or not key:
        raise SystemExit(
            "Faltan SUPABASE_URL y SUPABASE_ANON_KEY (o SUPABASE_SERVICE_KEY) en .env"
        )
    return create_client(url, key)


def retrieve_chunks_vector(
    client,
    query_embedding: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    response = client.rpc(
        "match_chat_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": limit,
        },
    ).execute()
    rows = response.data or []
    for row in rows:
        row["_source"] = "vector"
        row["_score"] = float(row.get("similarity") or 0.0)
    return rows


def retrieve_chunks_keywords(
    client,
    keywords: list[str],
    limit_per_kw: int = 4,
) -> list[dict[str, Any]]:
    if not keywords:
        return []

    merged: dict[str, dict[str, Any]] = {}
    for kw in keywords:
        variants = keyword_variants(kw) if not re.search(r"\d", kw) else [kw]
        rows: list[dict[str, Any]] = []

        for variant in variants:
            try:
                response = client.rpc(
                    "search_chat_chunks_text",
                    {"search_query": variant, "match_count": limit_per_kw},
                ).execute()
                rows.extend(response.data or [])
            except Exception:
                response = (
                    client.table("chat_chunks")
                    .select("id, content, partner_messages")
                    .ilike("content", f"%{variant}%")
                    .limit(limit_per_kw)
                    .execute()
                )
                rows.extend(response.data or [])

        for row in rows:
            chunk_id = row["id"]
            base = 0.72 if re.search(r"\d", kw) else 0.58
            score = float(row.get("similarity") or base)
            existing = merged.get(chunk_id)
            if not existing or score > existing["_score"]:
                merged[chunk_id] = {
                    **row,
                    "_source": "keyword",
                    "_score": score,
                    "_keyword": kw,
                }

    return list(merged.values())


def merge_chunk_results(
    vector_rows: list[dict[str, Any]],
    keyword_rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for row in vector_rows:
        merged[row["id"]] = {**row}

    for row in keyword_rows:
        chunk_id = row["id"]
        if chunk_id in merged:
            merged[chunk_id]["_source"] = "vector+keyword"
            merged[chunk_id]["_score"] = max(
                float(merged[chunk_id].get("_score") or 0.0),
                float(row.get("_score") or 0.0) + 0.12,
            )
        else:
            merged[chunk_id] = {**row}

    ranked = sorted(
        merged.values(),
        key=lambda r: float(r.get("_score") or 0.0),
        reverse=True,
    )
    return ranked[:top_k]


def retrieve_chunks(
    question: str,
    query_embedding: list[float],
    top_k: int,
    *,
    partner_name: str,
    my_name: str,
    vector_pool: int,
) -> list[dict[str, Any]]:
    client = get_supabase_client()
    vector_rows = retrieve_chunks_vector(client, query_embedding, vector_pool)
    keywords = extract_keywords(question, partner_name, my_name)
    keyword_rows = retrieve_chunks_keywords(client, keywords)
    rows = merge_chunk_results(vector_rows, keyword_rows, top_k)

    if not rows:
        raise SystemExit(
            "No se recuperaron chunks. ¿Hay datos en chat_chunks? ¿Ejecutaste el SQL?"
        )
    return rows


def build_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        sim = chunk.get("similarity") or chunk.get("_score")
        sim_txt = f" (score={sim:.3f})" if isinstance(sim, (int, float)) else ""
        src = chunk.get("_source", "?")
        kw = chunk.get("_keyword")
        kw_txt = f" kw={kw}" if kw else ""
        parts.append(f"--- Fragmento {i} [{src}{kw_txt}]{sim_txt} ---\n{chunk['content']}")
    return "\n\n".join(parts)


def gemini_generate(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={urllib.parse.quote(api_key, safe='')}"
    )
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 1024,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Error de Gemini ({err.code}): {detail}") from err

    text = (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )
    if not text:
        raise SystemExit(f"Gemini no devolvió texto: {payload}")
    return text


def ask(
    question: str,
    *,
    model_name: str,
    gemini_model: str,
    top_k: int,
    partner_name: str,
    my_name: str,
    show_sources: bool,
    vector_pool: int,
) -> str:
    question = question.strip()
    if not question:
        raise SystemExit("Escribe una pregunta.")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("peg_aqui"):
        raise SystemExit(
            "Falta GEMINI_API_KEY en .env\n"
            "Obtén una en https://aistudio.google.com/apikey"
        )

    print("Buscando en el chat (vector + palabras clave)…", file=sys.stderr)
    query_embedding = encode_query(question, model_name)
    chunks = retrieve_chunks(
        question,
        query_embedding,
        top_k,
        partner_name=partner_name,
        my_name=my_name,
        vector_pool=vector_pool,
    )

    if show_sources:
        print("\n--- Fuentes recuperadas ---", file=sys.stderr)
        for chunk in chunks:
            score = chunk.get("similarity") or chunk.get("_score")
            sim_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
            src = chunk.get("_source", "?")
            kw = chunk.get("_keyword", "")
            kw_txt = f" · {kw}" if kw else ""
            preview = chunk["content"][:120].replace("\n", " ")
            print(
                f"  [{sim_txt}] {chunk['id']} ({src}{kw_txt}): {preview}…",
                file=sys.stderr,
            )
        print(file=sys.stderr)

    context = build_context(chunks)
    system_prompt = f"""Eres un asistente íntimo y respetuoso para una pareja. Respondes preguntas sobre {partner_name} usando SOLO los fragmentos del chat de WhatsApp entre {my_name} y {partner_name}.

Reglas:
- Responde siempre en español, con tono cercano pero honesto.
- Basa tus respuestas únicamente en el contexto proporcionado; no inventes hechos.
- Si la pregunta es sobre gustos, sentimientos, planes o detalles de {partner_name}, prioriza mensajes de {partner_name}.
- Si el contexto no alcanza, dilo con claridad.
- No cites números de fragmento; integra la información de forma natural.
- Respuestas concisas (máximo ~8 frases salvo que pidan detalle)."""

    user_prompt = f"Contexto del chat:\n{context}\n\nPregunta: {question}"

    print("Generando respuesta…", file=sys.stderr)
    return gemini_generate(api_key, system_prompt, user_prompt, gemini_model)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Preguntar sobre Claudia (RAG + Gemini)")
    parser.add_argument("question", nargs="?", help="Pregunta en texto")
    parser.add_argument("-i", "--interactive", action="store_true", help="Modo chat")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--vector-pool",
        type=int,
        default=DEFAULT_VECTOR_POOL,
        help="Candidatos vectoriales antes de fusionar con keywords",
    )
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    parser.add_argument("--partner-name", default=os.getenv("PARTNER_NAME", DEFAULT_PARTNER))
    parser.add_argument("--my-name", default=os.getenv("MY_NAME", DEFAULT_MY_NAME))
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Mostrar chunks recuperados en stderr",
    )
    args = parser.parse_args()

    def run_one(q: str) -> None:
        answer = ask(
            q,
            model_name=args.model,
            gemini_model=args.gemini_model,
            top_k=args.top_k,
            partner_name=args.partner_name,
            my_name=args.my_name,
            show_sources=args.show_sources,
            vector_pool=args.vector_pool,
        )
        print(answer)

    if args.interactive:
        print("RAG chat (vacío o 'salir' para terminar)\n")
        while True:
            try:
                q = input("Pregunta> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"salir", "exit", "quit", "q"}:
                break
            print()
            run_one(q)
            print()
        return

    if not args.question:
        parser.error("Indica una pregunta o usa -i")

    run_one(args.question)


if __name__ == "__main__":
    main()
