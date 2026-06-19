-- Búsqueda híbrida opcional: full-text en español para nombres propios y hechos concretos.
-- Ejecutar en Supabase SQL Editor si quieres la RPC (ask_partner.py también busca por ilike).

create extension if not exists pg_trgm;

create index if not exists chat_chunks_content_trgm_idx
  on chat_chunks using gin (content gin_trgm_ops);

create or replace function search_chat_chunks_text(
  search_query text,
  match_count int default 8
)
returns table (
  id text,
  content text,
  partner_messages int,
  similarity float
)
language sql
stable
as $$
  select
    c.id,
    c.content,
    c.partner_messages,
    similarity(c.content, search_query)::float as similarity
  from chat_chunks c
  where c.content ilike '%' || search_query || '%'
     or similarity(c.content, search_query) > 0.08
  order by similarity(c.content, search_query) desc
  limit match_count;
$$;

grant execute on function search_chat_chunks_text(text, int) to anon;
