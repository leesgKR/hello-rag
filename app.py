"""AX RAG 스타터: 문서를 넣으면 pgvector에 저장하고, 질문하면 관련 내용을 찾아 답한다."""
import os

import psycopg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

try:
    for line in open(".env", encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)
except FileNotFoundError:
    pass

client = OpenAI()
CHAT = os.getenv("AX_CHAT_MODEL", "ax-local")
EMBED = os.getenv("AX_EMBED_MODEL", "ax-embed")
DIM = int(os.getenv("EMBED_DIM", "1024"))      # bge-m3 = 1024
DB = os.environ["DATABASE_URL"]
# Qwen3.8은 기본으로 "생각(thinking)" 모드라 느리다. 일반 답변은 끄고, 필요하면 AX_THINKING=true
EXTRA = {"chat_template_kwargs": {"enable_thinking": os.getenv("AX_THINKING", "false").lower() == "true"}}
app = FastAPI()


def vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def embed(texts: list[str]) -> list[list[float]]:
    r = client.embeddings.create(model=EMBED, input=texts)
    return [d.embedding for d in r.data]


def chunks(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = text.strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


@app.on_event("startup")
def init():
    with psycopg.connect(DB, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(f"""CREATE TABLE IF NOT EXISTS docs(
            id bigserial PRIMARY KEY, title text, chunk text, embedding vector({DIM}))""")
        c.execute("CREATE INDEX IF NOT EXISTS docs_hnsw ON docs USING hnsw (embedding vector_cosine_ops)")


class Doc(BaseModel):
    title: str
    text: str


class Ask(BaseModel):
    question: str


@app.post("/api/ingest")
def ingest(d: Doc):
    parts = chunks(d.text)
    with psycopg.connect(DB) as c:
        for i in range(0, len(parts), 32):
            batch = parts[i:i + 32]
            for p, e in zip(batch, embed(batch)):
                c.execute("INSERT INTO docs(title, chunk, embedding) VALUES (%s, %s, %s::vector)", (d.title, p, vec(e)))
    return {"chunks": len(parts)}


@app.post("/api/ask")
def ask(a: Ask):
    q = embed([a.question])[0]
    with psycopg.connect(DB) as c:
        rows = c.execute("SELECT title, chunk FROM docs ORDER BY embedding <=> %s::vector LIMIT 4",
                         (vec(q),)).fetchall()
    context = "\n\n".join(f"[{t}]\n{ch}" for t, ch in rows)
    msgs = [{"role": "system", "content": "아래 자료만 근거로 한국어로 답하세요. 자료에 없으면 모른다고 답하세요. 근거 문서 제목을 함께 적으세요."},
            {"role": "user", "content": f"자료:\n{context}\n\n질문: {a.question}"}]
    r = client.chat.completions.create(model=CHAT, messages=msgs, temperature=0.3, extra_body=EXTRA)
    return {"answer": r.choices[0].message.content, "sources": sorted({t for t, _ in rows})}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AX 문서 Q&A</title>
<style>body{font-family:sans-serif;max-width:760px;margin:24px auto;padding:0 16px}textarea,input{width:100%;padding:8px;margin:4px 0}
section{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}#ans{white-space:pre-wrap}</style></head><body>
<h2>AX 문서 Q&A</h2>
<section><h3>1. 문서 넣기</h3><input id="t" placeholder="문서 제목"><textarea id="x" rows="6" placeholder="본문 붙여넣기"></textarea>
<button onclick="ing()">저장</button> <span id="s"></span></section>
<section><h3>2. 질문하기</h3><input id="q" placeholder="질문"><button onclick="ask()">질문</button><p id="ans"></p></section>
<script>
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
async function ing(){s.textContent='저장 중...';const j=await post('/api/ingest',{title:t.value,text:x.value});s.textContent=j.chunks+'개 조각 저장'}
async function ask(){ans.textContent='찾는 중...';const j=await post('/api/ask',{question:q.value});ans.textContent=j.answer+'\\n\\n근거: '+(j.sources||[]).join(', ')}
</script></body></html>"""
