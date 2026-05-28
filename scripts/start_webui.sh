#!/bin/bash
# 启动 LLaMA-Factory Web UI，确保 CUDA 环境正确

# 激活虚拟环境
source ~/sglang_env/bin/activate

# 设置 CUDA 环境变量
export CUDA_VISIBLE_DEVICES=0
export PATH="$HOME/sglang_env/bin:$PATH"

# 验证 CUDA
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"

# 启动 Web UI
cd ~/LLaMA-Factory
python src/webui.py
