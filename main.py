import argparse

from config.model_config import ModelConfig
from models.hybrid_model import HybridModel
from models.model import Model
from report.export import dump_result_json
import os


def main(args):
    config = ModelConfig(args.config_path)

    print("\n{s:{c}^{n}}".format(s=" Simulator Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Device type:", args.device_type))
    print("{:<40} {:<10}".format("World size:", args.world_size))
    print("{:<40} {:<10}".format("Attn type:", config.attn_type))
    print("{:<40} {:<10}".format("Use FP8 GEMM:", args.use_fp8_gemm))
    print("{:<40} {:<10}".format("Use FP8 KV:", args.use_fp8_kv))

    if config.is_hybrid_linear:
        model = HybridModel(args, config)
    else:
        model = Model(args, config)

    weights = model.print_weights_info()
    kvcache = model.print_kvcache_info()
    flops = model.print_flops_info()

    prefill = None
    if not args.decode_only:
        prefill = model.prefill()

    decode = None
    if not args.prefill_only:
        decode = model.decoding()

    if args.output_json:
        payload = {
            "meta": {
                "config_path": args.config_path,
                "device_type": args.device_type,
                "world_size": args.world_size,
                "num_nodes": args.num_nodes,
                "max_prefill_tokens": args.max_prefill_tokens,
                "decode_bs": args.decode_bs,
                "target_tgs": args.target_tgs,
                "target_tpot_ms": args.target_tpot,
                "target_isl": args.target_isl,
                "target_osl": args.target_osl,
                "use_fp8_gemm": bool(args.use_fp8_gemm),
                "use_fp8_kv": bool(args.use_fp8_kv),
                "enable_deepep": bool(args.enable_deepep),
                "enable_tbo": bool(args.enable_tbo),
                "sm_ratio": args.sm_ratio,
                "prefill_only": bool(args.prefill_only),
                "decode_only": bool(args.decode_only),
            },
            "model": {
                "attn_type": getattr(config, "attn_type", None),
                "model_name": getattr(config, "modelName", None),
                "is_hybrid_linear": bool(getattr(config, "is_hybrid_linear", False)),
            },
            "weights": weights,
            "kvcache": kvcache,
            "flops": flops,
            "prefill": prefill,
            "decode": decode,
        }
        model_name = getattr(config, "modelName", "unknown")
        mode = "prefill" if args.prefill_only else "decode"
        if mode == "prefill":
            filename = (
                f"{model_name}_"
                f"{mode}_"
                f"i{args.target_isl}_"
                f"o{args.target_osl}_"
                f"mt{args.max_prefill_tokens}.json"
            )
        else:
            filename = (
                f"{model_name}_"
                f"{mode}_"
                f"i{args.target_isl}_"
                f"o{args.target_osl}_"
                f"bs{args.decode_bs}.json"
            )
        output_path = os.path.join(args.output_json, filename)
        dump_result_json(output_path, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="H20",
        choices=["H20", "H800", "H200", "GB200"],
        help="Device type",
    )
    parser.add_argument("--world-size", type=int, default=1, help="Num of GPUs")
    parser.add_argument("--num-nodes", type=int, default=1, help="Num of nodes")
    parser.add_argument(
        "--max-prefill-tokens", type=int, default=4096, help="Max prefill tokens"
    )
    parser.add_argument(
        "--decode-bs",
        type=int,
        help="Decoding batchsize. If not specified, bs = tgs * tpot.",
    )
    parser.add_argument(
        "--target-tgs", type=float, default=2560, help="Target tokens/s per GPU"
    )
    parser.add_argument("--target-tpot", type=float, default=50, help="TPOT in ms")
    parser.add_argument(
        "--target-isl", type=int, default=4096, help="Input sequence length, in tokens"
    )
    parser.add_argument(
        "--target-osl", type=int, default=2048, help="Output sequence length, in tokens"
    )
    parser.add_argument("--use-fp8-gemm", action="store_true", help="Use fp8 gemm")
    parser.add_argument("--use-fp8-kv", action="store_true", help="Use fp8 kvcache")
    parser.add_argument("--enable-deepep", action="store_true", help="Enable DeepEP")
    parser.add_argument(
        "--enable-tbo", action="store_true", help="Enable two batch overlap"
    )
    parser.add_argument(
        "--sm-ratio",
        type=float,
        default=108 / 132,
        help="In TBO DeepEP normal mode, the SM ratio used for computation",
    )
    parser.add_argument(
        "--prefill-only", action="store_true", help="Only simulate prefill"
    )
    parser.add_argument(
        "--decode-only", action="store_true", help="Only simulate decoding"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, export structured results to this JSON path (for golden tests).",
    )
    args = parser.parse_args()
    main(args)
