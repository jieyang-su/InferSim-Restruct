#!/bin/bash

python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --device-type H800 \
  --world-size 32 --num-nodes 4 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --max-prefill-tokens 8192 --prefill-only

python3 main.py --config-path hf_configs/deepseek_v3.2_config.json  \
  --device-type H800 \
  --world-size 32 --num-nodes 4 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --max-prefill-tokens 8192 --prefill-only

python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --device-type H800 \
  --world-size 32 --num-nodes 4 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --max-prefill-tokens 16384 --prefill-only \
  --target-isl 8192

python3 main.py --config-path hf_configs/deepseek_v3.2_config.json  \
  --device-type H800 \
  --world-size 32 --num-nodes 4 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --max-prefill-tokens 16384 --prefill-only \
  --target-isl 8192

python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --device-type H800 \
  --world-size 32 --num-nodes 4 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --max-prefill-tokens 32768 --prefill-only \
  --target-isl 16384

python3 main.py --config-path hf_configs/deepseek_v3.2_config.json  \
  --device-type H800 \
  --world-size 32 --num-nodes 4 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --max-prefill-tokens 32768 --prefill-only \
  --target-isl 16384
