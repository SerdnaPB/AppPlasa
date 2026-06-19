/**
 * RAG en el navegador: embedding local (Transformers.js) + Supabase + Gemini.
 */
(function (global) {
  const EMBED_MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
  const TRANSFORMERS_CDN =
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0";
  const DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite";
  const DEFAULT_TOP_K = 12;
  const DEFAULT_VECTOR_POOL = 20;
  const DEFAULT_PARTNER = "Claudia";
  const DEFAULT_MY_NAME = "Andres";

  const STOPWORDS = new Set(
    `
    quien quienes como cuando donde cual cuales que qué sobre con para por del
    de la las los una uno unos unas tiene tienen haber esta este estos esas
    claudia andres musica música gusta gustan cosas algo muy mas más
    quedado quedó quedo junio julio agosto septiembre octubre noviembre diciembre
    enero febrero marzo abril mayo menciona mencionar
    `.trim().split(/\s+/)
  );

  const MONTHS = {
    enero: 1,
    febrero: 2,
    marzo: 3,
    abril: 4,
    mayo: 5,
    junio: 6,
    julio: 7,
    agosto: 8,
    septiembre: 9,
    octubre: 10,
    noviembre: 11,
    diciembre: 12,
  };

  let embedderPromise = null;
  let supabaseClient = null;
  let partnerName = DEFAULT_PARTNER;
  let myName = DEFAULT_MY_NAME;
  let geminiModel = DEFAULT_GEMINI_MODEL;
  let onStatus = () => {};

  function stripAccents(value) {
    return value.normalize("NFD").replace(/\p{M}/gu, "");
  }

  function extractDatePatterns(question) {
    const patterns = [];
    const lower = question.toLocaleLowerCase("es");

    const slash = /\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b/g;
    let match;
    while ((match = slash.exec(question))) {
      const day = Number(match[1]);
      const month = Number(match[2]);
      const year = match[3];
      if (year) patterns.push(`${day}/${month}/${String(year).slice(-2)}`);
      patterns.push(`${day}/${month}`);
    }

    const spoken =
      /\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b/gi;
    while ((match = spoken.exec(lower))) {
      const day = Number(match[1]);
      const month = MONTHS[match[2].toLowerCase()];
      patterns.push(`${day}/${month}`);
      patterns.push(`${day}/${month}/26`);
    }

    return [...new Set(patterns)].slice(0, 4);
  }

  function keywordVariants(word) {
    const variants = new Set([word, word.toLocaleLowerCase("es"), stripAccents(word).toLocaleLowerCase("es")]);
    return [...variants].filter((v) => v.length >= 3);
  }

  function extractKeywords(question) {
    const words = question.match(/[\p{L}\p{N}_áéíóúñÁÉÍÓÚÑ]+/gu) || [];
    const skip = new Set([...STOPWORDS, partnerName.toLocaleLowerCase("es"), myName.toLocaleLowerCase("es")]);
    const keywords = [];

    for (const word of words) {
      const key = word.toLocaleLowerCase("es");
      if (key.length < 3 || skip.has(key)) continue;
      if (!keywords.includes(word)) keywords.push(word);
      if (keywords.length >= 6) break;
    }

    keywords.push(...extractDatePatterns(question));
    return keywords.slice(0, 8);
  }

  async function loadEmbedder() {
    if (!embedderPromise) {
      embedderPromise = (async () => {
        onStatus("Cargando modelo de búsqueda (solo la primera vez)…");
        const { pipeline, env } = await import(TRANSFORMERS_CDN);
        env.allowLocalModels = false;
        env.useBrowserCache = true;
        return pipeline("feature-extraction", EMBED_MODEL_ID);
      })();
    }
    return embedderPromise;
  }

  async function encodeQuery(text) {
    const model = await loadEmbedder();
    const tensor = await model(text, { pooling: "mean", normalize: true });
    return Array.from(tensor.data);
  }

  async function retrieveVector(embedding, limit) {
    const { data, error } = await supabaseClient.rpc("match_chat_chunks", {
      query_embedding: embedding,
      match_count: limit,
    });
    if (error) throw error;
    return (data || []).map((row) => ({
      ...row,
      _source: "vector",
      _score: Number(row.similarity || 0),
    }));
  }

  async function retrieveKeyword(keyword, limitPerKw = 4) {
    const variants = /\d/.test(keyword) ? [keyword] : keywordVariants(keyword);
    const merged = new Map();

    for (const variant of variants) {
      let rows = [];
      const { data, error } = await supabaseClient.rpc("search_chat_chunks_text", {
        search_query: variant,
        match_count: limitPerKw,
      });

      if (!error && data) {
        rows = data;
      } else {
        const res = await supabaseClient
          .from("chat_chunks")
          .select("id, content, partner_messages")
          .ilike("content", `%${variant}%`)
          .limit(limitPerKw);
        if (res.error) throw res.error;
        rows = res.data || [];
      }

      for (const row of rows) {
        const base = /\d/.test(keyword) ? 0.72 : 0.58;
        const score = Number(row.similarity || base);
        const prev = merged.get(row.id);
        if (!prev || score > prev._score) {
          merged.set(row.id, {
            ...row,
            _source: "keyword",
            _score: score,
            _keyword: keyword,
          });
        }
      }
    }

    return [...merged.values()];
  }

  async function retrieveChunks(question, embedding, topK, vectorPool) {
    const vectorRows = await retrieveVector(embedding, vectorPool);
    const keywords = extractKeywords(question);
    const keywordRows = [];

    for (const kw of keywords) {
      const rows = await retrieveKeyword(kw);
      keywordRows.push(...rows);
    }

    const merged = new Map();
    for (const row of vectorRows) merged.set(row.id, { ...row });

    for (const row of keywordRows) {
      const prev = merged.get(row.id);
      if (prev) {
        prev._source = "vector+keyword";
        prev._score = Math.max(prev._score, row._score + 0.12);
      } else {
        merged.set(row.id, { ...row });
      }
    }

    return [...merged.values()]
      .sort((a, b) => b._score - a._score)
      .slice(0, topK);
  }

  function buildContext(chunks) {
    return chunks
      .map((chunk, i) => `--- Fragmento ${i + 1} ---\n${chunk.content}`)
      .join("\n\n");
  }

  async function geminiGenerate(apiKey, systemPrompt, userPrompt) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${geminiModel}:generateContent?key=${encodeURIComponent(apiKey)}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemPrompt }] },
        contents: [{ role: "user", parts: [{ text: userPrompt }] }],
        generationConfig: { temperature: 0.35, maxOutputTokens: 1024 },
      }),
    });
    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload?.error?.message || `Gemini error ${res.status}`);
    }
    const text = payload?.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("").trim();
    if (!text) throw new Error("Gemini no devolvió texto.");
    return text;
  }

  function getConfig() {
    return global.RAG_CONFIG || {};
  }

  function getApiKey() {
    return String(getConfig().geminiApiKey || "").trim();
  }

  async function ask(question, options = {}) {
    const q = String(question || "").trim();
    if (!q) throw new Error("Escribe una pregunta.");

    const apiKey = options.apiKey || getApiKey();
    if (!apiKey) {
      throw new Error("Falta geminiApiKey en rag-config.js (copia rag-config.example.js).");
    }

    const topK = options.topK || DEFAULT_TOP_K;
    const vectorPool = options.vectorPool || DEFAULT_VECTOR_POOL;

    onStatus("Preparando búsqueda en el chat…");
    const embedding = await encodeQuery(q);

    onStatus("Buscando fragmentos relevantes…");
    const chunks = await retrieveChunks(q, embedding, topK, vectorPool);
    if (!chunks.length) throw new Error("No se encontraron fragmentos en Supabase.");

    const context = buildContext(chunks);
    const systemPrompt = `Eres un asistente íntimo y respetuoso para una pareja. Respondes preguntas sobre ${partnerName} usando SOLO los fragmentos del chat de WhatsApp entre ${myName} y ${partnerName}.

Reglas:
- Responde siempre en español, con tono cercano pero honesto.
- Basa tus respuestas únicamente en el contexto proporcionado; no inventes hechos.
- Si la pregunta es sobre gustos, sentimientos, planes o detalles de ${partnerName}, prioriza mensajes de ${partnerName}.
- Si el contexto no alcanza, dilo con claridad.
- No cites números de fragmento; integra la información de forma natural.
- Respuestas concisas (máximo ~8 frases salvo que pidan detalle).`;

    onStatus("Generando respuesta…");
    const answer = await geminiGenerate(
      apiKey,
      systemPrompt,
      `Contexto del chat:\n${context}\n\nPregunta: ${q}`
    );

    return { answer, chunks };
  }

  function initUi(root, config = {}) {
    if (!root) return;

    const ragConfig = getConfig();
    supabaseClient = config.supabaseClient;
    partnerName = config.partnerName || ragConfig.partnerName || DEFAULT_PARTNER;
    myName = config.myName || ragConfig.myName || DEFAULT_MY_NAME;
    geminiModel = config.geminiModel || ragConfig.geminiModel || DEFAULT_GEMINI_MODEL;
    onStatus = config.onStatus || (() => {});

    const statusEl = root.querySelector("#ragStatus");
    const messagesEl = root.querySelector("#ragMessages");
    const questionInput = root.querySelector("#ragQuestion");
    const askBtn = root.querySelector("#ragAskBtn");

    function setStatus(text, isError = false) {
      if (!statusEl) return;
      statusEl.textContent = text;
      statusEl.dataset.error = isError ? "1" : "0";
    }

    if (!getApiKey()) {
      setStatus("Falta rag-config.js con geminiApiKey.", true);
    }

    function appendMessage(role, text) {
      if (!messagesEl) return;
      const bubble = document.createElement("div");
      bubble.className = `rag-msg rag-msg-${role}`;
      bubble.textContent = text;
      messagesEl.appendChild(bubble);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function sendQuestion() {
      const q = questionInput?.value?.trim();
      if (!q) return;

      appendMessage("user", q);
      if (questionInput) questionInput.value = "";
      askBtn.disabled = true;
      setStatus("Trabajando…");

      try {
        onStatus = setStatus;
        const { answer } = await ask(q);
        appendMessage("assistant", answer);
        setStatus("Listo.");
      } catch (err) {
        const msg = err?.message || String(err);
        appendMessage("system", msg);
        setStatus(msg, true);
      } finally {
        askBtn.disabled = false;
        questionInput?.focus();
      }
    }

    askBtn?.addEventListener("click", sendQuestion);
    questionInput?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
      }
    });
  }

  global.RagAsk = {
    ask,
    initUi,
    getApiKey,
  };
})(window);
