-- Ejecutar en Supabase → SQL Editor (una vez).
-- Embeddings locales: paraphrase-multilingual-MiniLM-L12-v2 → 384 dimensiones.

create extension if not exists vector;

create table if not exists chat_chunks (
  id text primary key,
  content text not null,
  embedding vector(384) not null,
  embedding_model text not null default 'paraphrase-multilingual-MiniLM-L12-v2',
  message_count int,
  char_count int,
  date_start text,
  time_start text,
  date_end text,
  time_end text,
  partner_messages int,
  start_index int,
  end_index int,
  created_at timestamptz not null default now()
);

-- Búsqueda por similitud coseno (para consultas RAG).
create or replace function match_chat_chunks(
  query_embedding vector(384),
  match_count int default 6
)
returns table (
  id text,
  content text,
  similarity float,
  partner_messages int
)
language sql
stable
as $$
  select
    c.id,
    c.content,
    1 - (c.embedding <=> query_embedding) as similarity,
    c.partner_messages
  from chat_chunks c
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- Índice vectorial: créalo DESPUÉS de cargar los datos con embed_chunks.py
-- create index if not exists chat_chunks_embedding_idx
--   on chat_chunks using ivfflat (embedding vector_cosine_ops)
--   with (lists = 100);

-- Lectura para la app (anon). Escritura solo con service_role desde el script local.
alter table chat_chunks enable row level security;

drop policy if exists "chat_chunks_select_anon" on chat_chunks;
create policy "chat_chunks_select_anon"
  on chat_chunks for select
  to anon
  using (true);

-- No policy de INSERT/UPDATE para anon → solo service_role puede subir chunks.

grant usage on schema public to anon;
grant select on chat_chunks to anon;
grant execute on function match_chat_chunks(vector, int) to anon;
