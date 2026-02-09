from flops.flops import gemm_flops
from hardware.gpu import gpu_map
from layers.attn import get_gemm_mfu_and_latency
from mfu.mfu import (get_gemm_mfu, get_groupedgemm_decode_mfu,
                     get_groupedgemm_prefill_mfu)
from params.params import load_moe_weights_time


class MoE:
    """
    MoE/FFN layer, dense FFN is treated as a special 1-expert MoE
    """

    def __init__(self, config, use_fp8_gemm):
        self.use_fp8_gemm = use_fp8_gemm
        self.config = config

    def decode_moe(self, bs, device_type, num_gpus):
        gpu = gpu_map[device_type]

        routed_experts_gflops = gemm_flops(
            1, self.config.hidden_size, self.config.intermediate_size
        )
        routed_experts_gflops *= bs * self.config.num_experts_per_tok * 3.0 / 1e9

        moe_load_time = load_moe_weights_time(
            self.config, self.use_fp8_gemm, gpu, num_gpus
        ) * self.config.num_hidden_layers

        if self.config.is_moe:
            
            if self.config.modelPrefix == "DeepseekV3":
                routed_experts_mfu = max(
                    get_groupedgemm_decode_mfu(
                        self.config, bs, device_type, num_gpus, self.use_fp8_gemm
                    )
                ) * self.config.moe_layers 
                + get_gemm_mfu(
                    device_type,
                    bs,
                    self.config.hidden_size,
                    self.config.intermediate_size * 2 // num_gpus,
                ) * self.config.ffn_layers 
            else:
                routed_experts_mfu = max(
                    get_groupedgemm_decode_mfu(
                        self.config, bs, device_type, num_gpus, self.use_fp8_gemm
                    )
                ) * self.config.moe_layers 

        else:  # Dense FFN is treated as a special 1-expert MoE
            routed_experts_mfu = get_gemm_mfu(
                device_type,
                bs,
                self.config.hidden_size,
                self.config.intermediate_size * 2 // num_gpus,
            ) * self.config.num_hidden_layers
        if self.use_fp8_gemm:
            tflops = gpu.fp8_tflops
        else:
            tflops = gpu.fp16_tflops

        routed_experts_latency = routed_experts_gflops / (
            tflops * 1024 * routed_experts_mfu
        )
        
        print("{:<40} {:<10.2f}".format("Routed experts/FFN MFU:", routed_experts_mfu/self.config.num_hidden_layers))
        print(
            "{:<40} {:<10.2f}".format(
                "Routed experts/FFN latency (us):", (routed_experts_latency/self.config.num_hidden_layers) * 1e6
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Experts loading latency (us):", (moe_load_time/self.config.num_hidden_layers) * 1e6
            )
        )
        t = max(routed_experts_latency, moe_load_time)

        if self.config.num_shared_experts > 0:
            shared_expert_up_proj = get_gemm_mfu_and_latency(
                m=bs,
                k=self.config.hidden_size,
                n=self.config.intermediate_size * 2 * self.config.num_shared_experts,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )

            shared_expert_down_proj = get_gemm_mfu_and_latency(
                m=bs,
                k=self.config.intermediate_size * self.config.num_shared_experts,
                n=self.config.hidden_size,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Shared expert latency (us):",
                    (shared_expert_up_proj + shared_expert_down_proj) * 1e6,
                )
            )
            t += (shared_expert_up_proj + shared_expert_down_proj) * self.config.num_hidden_layers
        return t

    def prefill_moe(self, seq_len, device_type, num_gpus):
        gpu = gpu_map[device_type]

        routed_experts_gflops = gemm_flops(
            1, self.config.hidden_size, self.config.intermediate_size
        )
        routed_experts_gflops *= seq_len * self.config.num_experts_per_tok * 3.0 / 1e9

        moe_load_time = load_moe_weights_time(
            self.config, self.use_fp8_gemm, gpu, num_gpus
        ) * self.config.num_hidden_layers

        routed_ffn_mfu = 0.

        if self.config.is_moe:
            if (self.config.modelPrefix == "DeepseekV3"):
                routed_experts_mfu = max(
                    get_groupedgemm_prefill_mfu(
                        self.config, seq_len, device_type, num_gpus, self.use_fp8_gemm
                    )
                ) 
                routed_ffn_mfu= get_gemm_mfu(
                    device_type,
                    seq_len,
                    self.config.hidden_size,
                    self.config.intermediate_size * 2 // num_gpus,
                )
            else:
                routed_experts_mfu = max(
                    get_groupedgemm_prefill_mfu(
                        self.config, seq_len, device_type, num_gpus, self.use_fp8_gemm
                    )
                )

        else:  # Dense FFN is treated as a special 1-expert MoE
            routed_experts_mfu = get_gemm_mfu(
                device_type,
                seq_len,
                self.config.hidden_size,
                self.config.intermediate_size * 2 // num_gpus,
            )

        if self.use_fp8_gemm:
            tflops = gpu.fp8_tflops
        else:
            tflops = gpu.fp16_tflops
        if (self.config.modelPrefix == "DeepseekV3" and self.config.is_moe):
            routed_experts_latency = (routed_experts_gflops / (tflops * 1024 * routed_experts_mfu
                                                              ) * self.config.moe_layers) + (routed_experts_gflops / (
                                                                  tflops * 1024 * routed_ffn_mfu) * self.config.ffn_layers)
        else:
            routed_experts_latency = routed_experts_gflops / (
                tflops * 1024 * routed_experts_mfu
            ) * self.config.num_hidden_layers
        

        print("{:<40} {:<10.2f}".format("Routed experts MFU:", (routed_experts_mfu)))
        print(
            "{:<40} {:<10.2f}".format(
                "Routed experts latency (us):", (routed_experts_latency/self.config.num_hidden_layers) * 1e6
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Experts loading latency (us):", (moe_load_time/self.config.num_hidden_layers) * 1e6
            )
        )
        t = max(routed_experts_latency, moe_load_time)

        if self.config.num_shared_experts > 0:
            shared_expert_up_proj = get_gemm_mfu_and_latency(
                m=seq_len,
                k=self.config.hidden_size,
                n=self.config.intermediate_size * 2 * self.config.num_shared_experts,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )

            shared_expert_down_proj = get_gemm_mfu_and_latency(
                m=seq_len,
                k=self.config.intermediate_size * self.config.num_shared_experts,
                n=self.config.hidden_size,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Shared expert latency (us):",
                    (shared_expert_up_proj + shared_expert_down_proj) * 1e6,
                )
            )
            t += (shared_expert_up_proj + shared_expert_down_proj) * self.config.num_hidden_layers
        return t
