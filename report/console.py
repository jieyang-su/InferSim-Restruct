from __future__ import annotations

from typing import Any, Dict, Optional


def _banner(title: str, char: str = "=") -> str:
    return "\n" + "{s:{c}^{n}}".format(s=f" {title} ", n=50, c=char)


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}"


def _fmt_us(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}"


def print_report(payload: Dict[str, Any]) -> None:
    """Render structured engine outputs to console.

    This is intentionally the only place with printing.
    """

    meta = payload.get("meta", {})
    model = payload.get("model", {})
    weights = payload.get("weights", {})
    kvcache = payload.get("kvcache", {})
    flops = payload.get("flops", {})

    print(_banner("Simulator Result", "="))
    print("{:<40} {:<10}".format("Device type:", meta.get("device_type")))
    print("{:<40} {:<10}".format("World size:", meta.get("world_size")))
    print("{:<40} {:<10}".format("Attn type:", model.get("attn_type")))
    print("{:<40} {:<10}".format("Use FP8 GEMM:", meta.get("use_fp8_gemm")))
    print("{:<40} {:<10}".format("Use FP8 KV:", meta.get("use_fp8_kv")))
    print("{:<40} {:<10}".format("Enable DeepEP:", meta.get("enable_deepep")))
    print("{:<40} {:<10}".format("Enable TBO:", meta.get("enable_tbo")))

    print(_banner("Model Weights", "-"))
    if "full_attn_params_mb" in weights:
        print(
            "{:<40} {:<10.2f}".format(
                "One full attn params size (MB):", weights.get("full_attn_params_mb", 0.0)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One linear attn params size (MB):", weights.get("linear_attn_params_mb", 0.0)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", weights.get("expert_params_mb", 0.0)
            )
        )
    else:
        print(
            "{:<40} {:<10.2f}".format(
                "One attn params size (MB):", weights.get("attn_params_mb", 0.0)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", weights.get("expert_params_mb", 0.0)
            )
        )
    print(
        "{:<40} {:<10.2f}".format("Per GPU params size (GB):", weights.get("params_per_gpu_gb", 0.0))
    )
    print(
        "{:<40} {:<10.2f}".format(
            "KV cache budget (GB):", weights.get("kvcache_budget_gb", 0.0)
        )
    )

    print(_banner("KV Cache", "-"))
    print(
        "{:<40} {:<10.2f}".format("KV cache budget (GB):", kvcache.get("kvcache_budget_gb", 0.0))
    )
    print("{:<40} {:<10}".format("Input seq len:", kvcache.get("input_seq_len")))
    print("{:<40} {:<10}".format("Output seq len:", kvcache.get("output_seq_len")))
    print("{:<40} {:<10}".format("Target decode batchsize:", kvcache.get("target_decode_bs")))
    if "target_per_req_cache_mb" in kvcache:
        # HybridLinear style accounting
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-req cache size (MB):", kvcache.get("target_per_req_cache_mb", 0.0)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current full-attn KV (MB):", kvcache.get("current_full_attn_kv_mb", 0.0)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current states (MB):", kvcache.get("current_states_mb", 0.0)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current total cache (MB):", kvcache.get("current_total_cache_mb", 0.0)
            )
        )
    else:
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-token KV cache size (KB):", kvcache.get("target_per_token_kv_kb", 0.0)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-token KV cache size (KB):", kvcache.get("current_per_token_kv_kb", 0.0)
            )
        )
    if not kvcache.get("budget_ok", True):
        print("!Error: need smaller kvcache")

    print(_banner("FLOPs", "-"))
    print("{:<40} {:<10}".format("Num hidden layers:", flops.get("num_hidden_layers")))
    ppl = (flops.get("per_token_per_layer_gflops") or {})
    pt = (flops.get("per_token_gflops") or {})
    if "full_attn_core" in ppl:
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer full attn core (GFLOPs):", ppl.get("full_attn_core", 0.0)
            )
        )
    else:
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer attn core (GFLOPs):", ppl.get("attn_core", 0.0)
            )
        )
    print(
        "{:<40} {:<10.2f}".format("Per-token per-layer MoE/FFN (GFLOPs):", ppl.get("moe_ffn", 0.0))
    )
    print(
        "{:<40} {:<10.2f}".format("Per-token per-layer others (GFLOPs):", ppl.get("others", 0.0))
    )
    print(
        "{:<40} {:<10.2f}".format("Per-token total (GFLOPs):", pt.get("total", 0.0))
    )

    prefill = payload.get("prefill")
    if prefill is not None:
        print(_banner("Prefilling", "-"))
        print(
            "{:<40} {:<10}".format("Max prefill tokens:", prefill.get("max_prefill_tokens"))
        )
        br = prefill.get("breakdown", {})
        print(
            "{:<40} {:<10}".format(
                "attn_core_ms (ms):", _fmt_us(br.get("attn_core_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "kv_load_time_ms (ms):", _fmt_us(br.get("kv_load_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "indexer_core_ms (ms):", _fmt_us(br.get("indexer_core_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "indexer_kv_load_time_ms (ms):", _fmt_us(br.get("indexer_kv_load_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "attn_other_ms (ms):", _fmt_us(br.get("attn_other_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Routed_experts_ms (ms):", _fmt_us(br.get("moe_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Shared_experts_ms (ms):", _fmt_us(br.get("shared_expert_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Comm before MoE/FFN (us):", _fmt_us(br.get("comm_before_us"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Comm after MoE/FFN (us):", _fmt_us(br.get("comm_after_us"))
            )
        )
        print("{:<40} {:<10.2f}".format("TTFT (ms):", prefill.get("ttft_ms", 0.0)))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):", prefill.get("tgs_tok_per_gpu_s", 0.0)
            )
        )

    decode = payload.get("decode")
    if decode is not None:
        print(_banner("Decoding", "-"))
        br = decode.get("breakdown", {})
        print(
            "{:<40} {:<10}".format(
                "attn_core_ms (ms):", _fmt_us(br.get("attn_core_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "kv_load_time_ms (ms):", _fmt_us(br.get("kv_load_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "indexer_core_ms (ms):", _fmt_us(br.get("indexer_core_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "indexer_kv_load_time_ms (ms):", _fmt_us(br.get("indexer_kv_load_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "attn_other_ms (ms):", _fmt_us(br.get("attn_other_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Routed_experts_ms (ms):", _fmt_us(br.get("moe_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Shared_experts_ms (ms):", _fmt_us(br.get("shared_expert_ms"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Comm before MoE/FFN (us):", _fmt_us(br.get("comm_before_us"))
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Comm after MoE/FFN (us):", _fmt_us(br.get("comm_after_us"))
            )
        )
        print("{:<40} {:<10.2f}".format("TPOT (ms):", decode.get("tpot_ms", 0.0)))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS):", decode.get("tgs_tok_per_gpu_s", 0.0)
            )
        )
        if decode.get("slo_violation"):
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
