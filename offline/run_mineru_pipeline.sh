#!/bin/bash
# run_mineru_pipeline.sh — 使用MinerU解析指定KB的PDF并运行完整离线管道
# 用法:
#   bash offline/run_mineru_pipeline.sh              # 默认：招股说明书2
#   bash offline/run_mineru_pipeline.sh 招股说明书2   # 指定KB
#
# v2.0: KB_NAME通过命令行参数或环境变量指定

set -e

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- 解析KB名称 ----------
KB_NAME="${1:-招股说明书2}"
export KB_NAME

# ---------- 环境配置 ----------
export MINERU_MODEL_SOURCE=modelscope
export MINERU_PDF_RENDER_THREADS=1
export TOKENIZERS_PARALLELISM=false

MINERU_PYTHON="/home/qjx/miniconda3/envs/llama-factory/bin/python"

echo "=========================================="
echo " MinerU版离线管道 — KB: $KB_NAME"
echo "=========================================="
echo "MinerU:    $( $MINERU_PYTHON -c 'from mineru import version; print(version.__version__)' 2>/dev/null || echo '3.x' )"
echo "PDF:       data/${KB_NAME}.pdf"
echo "=========================================="

# ---------- 1. MinerU解析PDF ----------
echo ""
echo "[Step 1/3] MinerU解析PDF: $KB_NAME"
$MINERU_PYTHON offline/parse_pdf_mineru.py "$KB_NAME"

# ---------- 2. 文本分块 ----------
echo ""
echo "[Step 2/3] 文本分块: $KB_NAME"
$MINERU_PYTHON offline/chunk_text.py "$KB_NAME"

# ---------- 3. 向量化 ----------
echo ""
echo "[Step 3/3] 向量化: $KB_NAME"
$MINERU_PYTHON offline/generate_embeddings.py "$KB_NAME"

echo ""
echo "=========================================="
echo "  KB '$KB_NAME' 离线管道执行完成！"
echo "  接下来请运行 import_to_milvus.py 导入向量库"
echo "  或运行 run_all_pipeline.sh 一次完成所有KB"
echo "=========================================="
