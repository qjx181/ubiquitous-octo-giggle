#!/bin/bash
# 导出合并后的完整模型

source ~/sglang_env/bin/activate

cd ~/LLaMA-Factory

# 导出模型
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3.5-0.8B-Base \
  --adapter_name_or_path saves/Qwen3.5-0.8B-Base/lora/train_2026-05-12-08-33-52 \
  --template qwen \
  --finetuning_type lora \
  --export_dir saves/qwen35-merged \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format false

echo "✅ 模型已导出到: saves/qwen35-merged"
