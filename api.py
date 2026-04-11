"""
api.py — FastAPI backend for the RAG Query UI
Run with:  uvicorn api:app --reload --port 8000

Requirements (add to your existing env):
    pip install fastapi uvicorn
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from embedding import get_embedding_function   # your existing module

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "chroma"
GOOGLE_API_KEY = "AIzaSyBSuvNI6ljPsYylEed1lu_TZ_r3iBwg18M"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Query API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML UI at /  (place rag_query_ui.html in the same folder)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("rag_query_ui.html")


# ── Schema ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query_text: str

class SourceResult(BaseModel):
    id: str
    file: str
    page: int
    chunk: str
    score: float
    preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResult]


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    query_text = req.query_text

    # 1. Retrieve from Chroma
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    # 2. Build prompt and call Gemini
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key="AIzaSyBSuvNI6ljPsYylEed1lu_TZ_r3iBwg18M",
    )
    answer = model.invoke(prompt).content

    # 3. Parse sources
    sources: list[SourceResult] = []
    for doc, score in results:
        chunk_id = doc.metadata.get("id", "unknown")
        parts = chunk_id.split(":")
        file        = parts[0] if len(parts) > 0 else "unknown"
        page_raw    = parts[1] if len(parts) > 1 else "0"
        chunk_index = parts[2] if len(parts) > 2 else "0"

        try:
            page_num = int(page_raw) + 1
        except ValueError:
            page_num = 0

        sources.append(SourceResult(
            id      = chunk_id,
            file    = file,
            page    = page_num,
            chunk   = chunk_index,
            score   = round(float(score), 4),
            preview = doc.page_content[:200],
        ))

    return QueryResponse(answer=answer, sources=sources)