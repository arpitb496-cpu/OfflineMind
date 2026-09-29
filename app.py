"""OfflineMind: private, offline document Q&A (local RAG)."""
import os, numpy as np, streamlit as st
from pypdf import PdfReader
from fastembed import TextEmbedding
from llama_cpp import Llama

MODEL_PATH = os.environ.get("GGUF_MODEL", "models/model.gguf")
# Prefer the Snapdragon NPU (QNN) when onnxruntime-qnn is installed; else CPU.
PROVIDERS = ["QNNExecutionProvider", "CPUExecutionProvider"]

@st.cache_resource
def load_models():
    try:
        emb = TextEmbedding("BAAI/bge-small-en-v1.5", providers=PROVIDERS)
    except Exception:
        emb = TextEmbedding("BAAI/bge-small-en-v1.5")
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, verbose=False)
    return emb, llm

def chunk(text, size=800, overlap=100):
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]

st.title("OfflineMind: Private Document Q&A")
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
q = st.text_input("Ask a question about your documents")

if files and q:
    emb, llm = load_models()
    chunks = []
    for f in files:
        for n, page in enumerate(PdfReader(f).pages, 1):
            chunks += [(f.name, n, c) for c in chunk(page.extract_text() or "")]
    vecs = np.array(list(emb.embed([c[2] for c in chunks])))
    qv = np.array(list(emb.embed([q])))[0]
    top = np.argsort(vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv) + 1e-9))[::-1][:4]
    ctx = "\n\n".join(chunks[i][2] for i in top)
    prompt = f"Answer using only the context.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
    out = llm(prompt, max_tokens=300)["choices"][0]["text"]
    st.write(out)
    st.caption("Sources: " + ", ".join(f"{chunks[i][0]} p.{chunks[i][1]}" for i in top))
