#!/bin/bash
# run_all_pipeline.sh — 运行完整离线管道（所有知识库）
# 用法: bash offline/run_all_pipeline.sh
#
# v2.0: 遍历config.py中定义的所有KB，依次执行：
#   解析（fitz/mineru按配置选择）→ 分块 → 向量化 → 导入Milvus

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="/home/qjx/miniconda3/envs/llama-factory/bin/python"
MINERU_PYTHON="$PYTHON"

echo "=========================================="
echo " 离线管道（全量）"
echo "=========================================="

# 获取所有知识库名称（从config.py的KNOWLEDGE_BASES字典读取）
KB_NAMES=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
from config import KNOWLEDGE_BASES
for k in KNOWLEDGE_BASES:
    print(k)
")

echo "知识库列表:"
echo "$KB_NAMES" | while read kb; do echo "  - $kb"; done
echo ""

# 遍历每个KB
echo "$KB_NAMES" | while read KB_NAME; do
    [ -z "$KB_NAME" ] && continue

    # 读取该KB的解析器类型
    PARSER=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
from config import KNOWLEDGE_BASES
print(KNOWLEDGE_BASES.get('$KB_NAME', {}).get('parser', 'fitz'))
")

    export KB_NAME
    export MINERU_MODEL_SOURCE=modelscope
    export MINERU_PDF_RENDER_THREADS=1
    export TOKENIZERS_PARALLELISM=false

    echo ""
    echo "=========================================="
    echo "  [KB] $KB_NAME (解析器: $PARSER)"
    echo "=========================================="

    # Step 1: 解析PDF
    echo "[1/4] 解析PDF: $KB_NAME"
    if [ "$PARSER" = "mineru" ]; then
        $MINERU_PYTHON offline/parse_pdf_mineru.py "$KB_NAME" || true
    else
        $PYTHON offline/parse_pdf.py "$KB_NAME" || true
    fi

    # Step 2: 分块
    echo "[2/4] 文本分块: $KB_NAME"
    $PYTHON offline/chunk_text.py "$KB_NAME" || true

    # Step 3: 向量化
    echo "[3/4] 向量化: $KB_NAME"
    export CUDA_VISIBLE_DEVICES=0
    $PYTHON offline/generate_embeddings.py "$KB_NAME" || true
done

# Step 4: 导入Milvus（所有KB一起导入）
echo ""
echo "=========================================="
echo "  [4/4] 导入Milvus（所有KB）"
echo "=========================================="
$PYTHON offline/import_to_milvus.py

echo ""
echo "=========================================="
echo "  🎉 全量管道执行完成！"
echo "  共 $(echo "$KB_NAMES" | wc -l) 个知识库已就绪"
echo "=========================================="
