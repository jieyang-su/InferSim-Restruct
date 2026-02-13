#!/bin/bash
python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 1024


python3 main.py --config-path hf_configs/deepseek_v3.2_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 1024


python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 2048


python3 main.py --config-path hf_configs/deepseek_v3.2_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 2048

python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 8192

python3 main.py --config-path hf_configs/deepseek_v3.2_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 8192


python3 main.py --config-path hf_configs/deepseek_v3_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 4096 \
  --output-json ./output-json/ds

python3 main.py --config-path hf_configs/deepseek_v3.2_config.json  \
  --device-type H800 \
  --world-size 128 --num-nodes 16 \
  --use-fp8-gemm  --enable-deepep \
  --enable-tbo  --target-osl 1786 \
  --decode-bs 64 \
  --decode-only \
  --target-isl 4096 \
  --output-json ./output-json/ds