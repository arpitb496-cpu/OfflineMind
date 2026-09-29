# OfflineMind: Private Document Q&A on Snapdragon

Ask questions about your own PDFs, fully offline. Local RAG with source citations.

## Setup
```bash
pip install -r requirements.txt
# Download a quantized GGUF model into models/model.gguf (or set GGUF_MODEL)
streamlit run app.py
```

## Architecture
PDF -> chunks -> ONNX embeddings -> cosine search -> local LLM -> answer + page citations.

## Snapdragon optimization
Embeddings run through ONNX Runtime, preferring the QNN execution provider (NPU) with CPU fallback.
Models come from Qualcomm AI Hub / open-source platforms.

## Benchmarks
| Device | Backend | Latency | Tokens/sec |
|---|---|---|---|
| (fill with your measured results) | NPU / CPU | | |

## Roadmap
Indian-language support, voice input, OCR for scanned notes.
