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
Run `python benchmark.py` (default backend) and `python benchmark.py cpu` on the target
Snapdragon-powered HP PC. Each run prints a row for the table below.

| Device | Backend | Embedding latency | LLM speed |
|---|---|---|---|
| Snapdragon HP PC | NPU (QNN) | Pending measurement | Pending measurement |
| Snapdragon HP PC | CPU | Pending measurement | Pending measurement |

Status: numbers are not yet measured. They will be added after testing on a Snapdragon device.

## Roadmap
Indian-language support, voice input, OCR for scanned notes.
