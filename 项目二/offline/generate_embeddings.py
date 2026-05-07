"""
generate_embeddings.py — 文本转向量
作用：加载BGE-M3模型，把chunks转成向量，文本与向量解耦存储
"""

import json
import shutil
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CHUNKS_JSONL_PATH, EMBEDDINGS_JSONL_PATH, CHUNKS_DIR,
    EMBEDDING_MODEL_PATH, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM,
    RUN_TIMESTAMP, setup_logger,
)

logger = setup_logger("generate_embeddings")


def check_file_exists(file_path: str) -> bool:
    """
    作用：检查输入文件是否存在
    原理：分块JSONL是输入，不存在则无法向量化
    逻辑：Path.exists()判断
    """
    p = Path(file_path)
    if not p.exists():
        logger.error(f"文件不存在: {file_path}")
        return False
    return True


def backup_previous_output(output_path: str):
    """
    作用：备份旧的向量文件，用于回滚
    原理：向量化可能覆盖旧文件，备份后可以恢复
    逻辑：文件存在则复制为 原名_备份_时间戳.后缀
    """
    p = Path(output_path)
    if p.exists():
        bn = f"{p.stem}_备份_{RUN_TIMESTAMP}{p.suffix}"
        shutil.copy2(p, p.parent / bn)
        logger.info(f"已备份旧文件 -> {p.parent / bn}")


def load_chunks(jsonl_path: str) -> list[dict]:
    """
    作用：加载分块后的JSONL文件
    原理：JSONL每行一条JSON记录，逐行读取
    逻辑：逐行解析JSON，解析失败的单条跳过，不中断
    返回：chunk字典列表
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"第{ln}行解析失败: {e}")
    return chunks


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    作用：加载BGE-M3模型，把所有chunks的文字转成向量
    原理：
      - BGE-M3是一个多语言嵌入模型，输出1024维向量
      - 语义相似的文本在高维空间中距离近
      - normalize_embeddings=True 使向量归一化，后续用内积=余弦相似度
    逻辑：
      1. 从本地路径加载BGE-M3模型
      2. 提取所有chunk的text字段
      3. 批量encode（自动用GPU）
      4. 校验len(chunks)==len(embeddings)，不一致则返回空
      5. 解耦存储：向量文件只存chunk_id+元数据，不存text
    返回：带向量的chunk列表（不含text字段）
    """
    if not chunks:
        logger.error("没有chunk数据")
        return []
    mp = str(EMBEDDING_MODEL_PATH)
    if not Path(mp).exists():
        logger.error(f"模型路径不存在: {mp}")
        return []
    logger.info("加载BGE-M3模型...")
    try:
        model = SentenceTransformer(mp)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return []
    logger.info(f"模型加载完成，输出维度: {EMBEDDING_DIM}")
    texts = [c["text"] for c in chunks]
    logger.info(f"共{len(texts)}个chunks，转向量...")
    try:
        embs = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
        )
    except Exception as e:
        logger.error(f"向量化失败: {e}")
        return []
    logger.info(f"向量化完成，维度: {embs.shape[1]}")
    # 校验：向量数必须等于chunk数
    if len(embs) != len(texts):
        logger.error(f"向量数不匹配: chunks {len(texts)}, embeddings {len(embs)}")
        return []
    if embs.shape[1] != EMBEDDING_DIM:
        logger.warning(f"维度不匹配: 期望{EMBEDDING_DIM}, 实际{embs.shape[1]}")
    # 解耦：向量文件只存元数据，不存text
    result = []
    ok = 0
    for c, v in zip(chunks, embs):
        try:
            result.append({
                "chunk_id": c["chunk_id"],
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "section": c.get("section"),
                "source_pdf": c.get("source_pdf"),
                "embedding": v.tolist(),
            })
            ok += 1
        except Exception as e:
            logger.warning(f"chunk {c.get('chunk_id','?')} 处理失败: {e}")
    logger.info(f"转向量完成: 成功{ok}条")
    return result


def save_embeddings(chunks: list[dict]):
    """
    作用：保存带向量的chunks为JSONL
    逻辑：逐行写入JSON，每行一条记录
    """
    op = Path(str(EMBEDDINGS_JSONL_PATH))
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info(f"已保存: {op}")


def print_quality_report(chunks: list[dict]):
    """
    作用：输出向量化质量报告
    原理：让用户知道向量化是否完整，维度是否正确
    逻辑：统计总chunk数、含向量的比例、向量维度集合
    """
    if not chunks:
        return
    total = len(chunks)
    with_emb = sum(1 for c in chunks if "embedding" in c)
    dims = set(len(c["embedding"]) for c in chunks if "embedding" in c)
    logger.info("=" * 50)
    logger.info("  向量化质量报告")
    logger.info("=" * 50)
    logger.info(f"总chunks: {total}")
    logger.info(f"含向量: {with_emb}/{total}")
    logger.info(f"向量维度: {dims}")


def main():
    """作用：程序入口，加载chunks→转向量→质量报告→保存"""
    logger.info("=" * 50)
    logger.info("  文本转向量 开始")
    logger.info("=" * 50)
    if not check_file_exists(str(CHUNKS_JSONL_PATH)):
        return
    backup_previous_output(str(EMBEDDINGS_JSONL_PATH))
    chunks = load_chunks(str(CHUNKS_JSONL_PATH))
    if not chunks:
        logger.error("未加载到数据")
        return
    logger.info(f"加载了{len(chunks)}个chunks")
    vecs = generate_embeddings(chunks)
    if not vecs:
        logger.error("向量化失败")
        return
    print_quality_report(vecs)
    save_embeddings(vecs)
    logger.info("🎉 转向量完成")


if __name__ == "__main__":
    main()
