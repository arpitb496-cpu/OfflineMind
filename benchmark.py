"""Measure embedding latency and LLM tokens/sec, then print a README table row.
Run once per backend on the target Snapdragon PC:
    python benchmark.py            # default providers (QNN NPU if available, else CPU)
    python benchmark.py cpu        # force CPU for comparison
"""
import os, sys, time, platform
from fastembed import TextEmbedding
from llama_cpp import Llama

force_cpu = len(sys.argv) > 1 and sys.argv[1].lower() == "cpu"
providers = ["CPUExecutionProvider"] if force_cpu else ["QNNExecutionProvider", "CPUExecutionProvider"]
label = "CPU" if force_cpu else "NPU (QNN) if available"

emb = TextEmbedding("BAAI/bge-small-en-v1.5", providers=providers)
texts = ["Sample paragraph for benchmarking document embeddings."] * 64
list(emb.embed(texts))  # warm-up
t = time.perf_counter(); list(emb.embed(texts)); emb_ms = (time.perf_counter() - t) * 1000 / len(texts)

llm = Llama(model_path=os.environ.get("GGUF_MODEL", "models/model.gguf"), n_ctx=2048, verbose=False)
t = time.perf_counter()
out = llm("Explain offline AI in two sentences.", max_tokens=128)
tps = out["usage"]["completion_tokens"] / (time.perf_counter() - t)

print(f"| {platform.processor() or platform.machine()} | {label} | {emb_ms:.1f} ms/chunk embed | {tps:.1f} tok/s |")
