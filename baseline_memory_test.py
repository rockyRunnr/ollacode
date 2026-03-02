#!/usr/bin/env python3
"""
Phase 1: Baseline — Ollama 정적 KV Cache 메모리/성능 측정

다양한 num_ctx 설정에서 qwen3.5:27b의 실제 메모리 사용량과 추론 성능을 측정.
32GB Mac Mini 환경에서 정적 할당의 한계를 정량적으로 확인.

Usage:
    python baseline_memory_test.py [--model qwen3.5:27b] [--output results.json]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import httpx
import psutil

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen3.5:27b"

# Test matrix: num_ctx values to measure
NUM_CTX_VALUES = [4096, 8192, 16384, 32768, 65536, 131072]

# Short prompt for minimal-usage tests
SHORT_PROMPT = "What is 2+2? Answer in one word."

# Medium prompt (~4K tokens worth)
MEDIUM_PROMPT = "Explain the concept of virtual memory in operating systems. " * 100


@dataclass
class MeasurementResult:
    """Single measurement result."""
    experiment_id: str
    model: str
    num_ctx: int
    prompt_length: int  # approx tokens
    system_total_ram_gb: float
    memory_before_load_gb: float
    memory_after_load_gb: float
    memory_during_inference_gb: float
    memory_model_only_gb: float  # after_load - before_load
    kv_cache_theoretical_gb: float  # calculated
    ttft_ms: float  # time to first token
    total_time_s: float
    tokens_generated: int
    tokens_per_second: float
    ollama_reported_eval_count: int
    ollama_reported_eval_duration_ms: float
    error: str = ""


def get_system_memory_gb() -> float:
    """Get current system memory usage in GB."""
    mem = psutil.virtual_memory()
    return round(mem.used / (1024**3), 2)


def get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    mem = psutil.virtual_memory()
    return round(mem.total / (1024**3), 2)


def get_ollama_process_rss_gb() -> float:
    """Get Ollama runner process RSS in GB."""
    total_rss = 0.0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = proc.info["name"] or ""
            if "ollama" in name.lower():
                total_rss += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total_rss / (1024**3), 2)


def unload_model(model: str):
    """Unload model from Ollama to free memory."""
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass
    time.sleep(3)


def calculate_theoretical_kv_cache_gb(num_ctx: int) -> float:
    """Calculate theoretical KV cache size for Qwen3.5:27b.

    Qwen3.5:27b: 16 full-attention layers, 4 KV heads, 128 head_dim, FP16
    KV per token = 2 * n_kv_heads * head_dim * n_layers * sizeof(float16)
                 = 2 * 4 * 128 * 16 * 2 = 32,768 bytes ≈ 32 KB per token

    Note: actual may differ due to quantized KV cache (q8_0 etc.)
    """
    bytes_per_token = 2 * 4 * 128 * 16 * 2  # 32KB
    total_bytes = bytes_per_token * num_ctx
    return round(total_bytes / (1024**3), 3)


def run_inference(model: str, prompt: str, num_ctx: int) -> dict:
    """Run a single inference and return timing + response data."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": 100,  # generate 100 tokens
            "temperature": 0.0,
        },
    }

    start_time = time.time()
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": str(e), "total_time": time.time() - start_time}

    total_time = time.time() - start_time

    return {
        "response": data.get("response", ""),
        "total_time": total_time,
        "eval_count": data.get("eval_count", 0),
        "eval_duration": data.get("eval_duration", 0) / 1e6,  # ns → ms
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_eval_duration": data.get("prompt_eval_duration", 0) / 1e6,
        "total_duration": data.get("total_duration", 0) / 1e6,
        "load_duration": data.get("load_duration", 0) / 1e6,
    }


def run_single_experiment(
    model: str, num_ctx: int, prompt: str, experiment_id: str
) -> MeasurementResult:
    """Run a single experiment: load model with specific num_ctx, measure everything."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_id}")
    print(f"  num_ctx={num_ctx:,}  prompt_len≈{len(prompt)//4} tokens")
    print(f"{'='*60}")

    total_ram = get_total_ram_gb()
    kv_theoretical = calculate_theoretical_kv_cache_gb(num_ctx)

    # 1. Unload any existing model
    print("  [1/5] Unloading previous model...")
    unload_model(model)
    time.sleep(2)

    # 2. Measure memory before loading
    mem_before = get_system_memory_gb()
    ollama_rss_before = get_ollama_process_rss_gb()
    print(f"  [2/5] Memory before load: {mem_before:.2f} GB (Ollama RSS: {ollama_rss_before:.2f} GB)")

    # 3. Load model with specific num_ctx (warmup inference)
    print(f"  [3/5] Loading model with num_ctx={num_ctx:,}...")
    warmup_result = run_inference(model, "Hello", num_ctx)
    if "error" in warmup_result:
        print(f"  ❌ Load failed: {warmup_result['error']}")
        return MeasurementResult(
            experiment_id=experiment_id, model=model, num_ctx=num_ctx,
            prompt_length=0, system_total_ram_gb=total_ram,
            memory_before_load_gb=mem_before, memory_after_load_gb=0,
            memory_during_inference_gb=0, memory_model_only_gb=0,
            kv_cache_theoretical_gb=kv_theoretical,
            ttft_ms=0, total_time_s=0, tokens_generated=0,
            tokens_per_second=0, ollama_reported_eval_count=0,
            ollama_reported_eval_duration_ms=0,
            error=warmup_result["error"],
        )

    time.sleep(2)
    mem_after_load = get_system_memory_gb()
    ollama_rss_after = get_ollama_process_rss_gb()
    print(f"  [4/5] Memory after load: {mem_after_load:.2f} GB (Ollama RSS: {ollama_rss_after:.2f} GB)")
    print(f"         Model+KV allocated: {mem_after_load - mem_before:.2f} GB")
    print(f"         Theoretical KV: {kv_theoretical:.3f} GB")

    # 4. Run actual inference
    print(f"  [5/5] Running inference...")
    result = run_inference(model, prompt, num_ctx)
    if "error" in result:
        return MeasurementResult(
            experiment_id=experiment_id, model=model, num_ctx=num_ctx,
            prompt_length=len(prompt) // 4, system_total_ram_gb=total_ram,
            memory_before_load_gb=mem_before, memory_after_load_gb=mem_after_load,
            memory_during_inference_gb=get_system_memory_gb(),
            memory_model_only_gb=mem_after_load - mem_before,
            kv_cache_theoretical_gb=kv_theoretical,
            ttft_ms=0, total_time_s=result["total_time"], tokens_generated=0,
            tokens_per_second=0, ollama_reported_eval_count=0,
            ollama_reported_eval_duration_ms=0,
            error=result["error"],
        )

    mem_during = get_system_memory_gb()
    tokens = result.get("eval_count", 0)
    eval_dur_ms = result.get("eval_duration", 0)
    tps = (tokens / (eval_dur_ms / 1000)) if eval_dur_ms > 0 else 0
    ttft = result.get("prompt_eval_duration", 0)

    print(f"         Tokens generated: {tokens}")
    print(f"         Tokens/s: {tps:.1f}")
    print(f"         TTFT: {ttft:.0f} ms")
    print(f"         Memory during inference: {mem_during:.2f} GB")

    return MeasurementResult(
        experiment_id=experiment_id,
        model=model,
        num_ctx=num_ctx,
        prompt_length=len(prompt) // 4,
        system_total_ram_gb=total_ram,
        memory_before_load_gb=mem_before,
        memory_after_load_gb=mem_after_load,
        memory_during_inference_gb=mem_during,
        memory_model_only_gb=round(mem_after_load - mem_before, 2),
        kv_cache_theoretical_gb=kv_theoretical,
        ttft_ms=ttft,
        total_time_s=result["total_time"],
        tokens_generated=tokens,
        tokens_per_second=round(tps, 2),
        ollama_reported_eval_count=result.get("eval_count", 0),
        ollama_reported_eval_duration_ms=eval_dur_ms,
    )


def print_summary_table(results: list[MeasurementResult]):
    """Print a summary table of all results."""
    print(f"\n{'='*90}")
    print(f"  BASELINE RESULTS SUMMARY — {results[0].model}")
    print(f"  System RAM: {results[0].system_total_ram_gb:.0f} GB")
    print(f"{'='*90}")
    print(f"  {'ID':<8} {'num_ctx':>8} {'Prompt':>7} {'Mem Used':>9} {'KV Theo':>8} {'tok/s':>7} {'TTFT':>8} {'Status'}")
    print(f"  {'-'*8} {'-'*8} {'-'*7} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    for r in results:
        status = "❌ " + r.error[:20] if r.error else "✅"
        print(
            f"  {r.experiment_id:<8} {r.num_ctx:>8,} {r.prompt_length:>5}t "
            f"{r.memory_model_only_gb:>7.2f}GB {r.kv_cache_theoretical_gb:>6.3f}GB "
            f"{r.tokens_per_second:>6.1f} {r.ttft_ms:>6.0f}ms {status}"
        )

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Baseline KV cache memory measurement")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to test")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer num_ctx values)")
    args = parser.parse_args()

    ctx_values = [4096, 32768, 131072] if args.quick else NUM_CTX_VALUES

    print(f"🧪 Baseline KV Cache Memory Test")
    print(f"   Model: {args.model}")
    print(f"   RAM: {get_total_ram_gb():.0f} GB")
    print(f"   Tests: {len(ctx_values) * 2} experiments")
    print(f"   num_ctx values: {ctx_values}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results: list[MeasurementResult] = []

    for num_ctx in ctx_values:
        # Test A: Short prompt (100 tokens) — shows KV cache overhead
        r = run_single_experiment(
            args.model, num_ctx, SHORT_PROMPT,
            experiment_id=f"B-{num_ctx//1024}K-S"
        )
        results.append(r)

        # Test B: Medium prompt (~4K tokens) — shows actual usage
        if num_ctx >= 8192:
            r = run_single_experiment(
                args.model, num_ctx, MEDIUM_PROMPT,
                experiment_id=f"B-{num_ctx//1024}K-M"
            )
            results.append(r)

    # Final cleanup
    unload_model(args.model)

    # Summary
    print_summary_table(results)

    # Save results
    if args.output:
        output_data = {
            "metadata": {
                "model": args.model,
                "system_ram_gb": get_total_ram_gb(),
                "timestamp": datetime.now().isoformat(),
                "ollama_version": "0.17.4",
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
