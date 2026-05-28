"""
generate_embeddings_parallel.py — 并行文本转向量脚本
====================================================
作用：加载BGE-M3模型，把chunks转成1024维向量，支持断点续传和并行编码
原理：
  1. BGE-M3是北京智源研究院开源的多语言Embedding模型
  2. 将文本映射到高维向量空间，语义相似的文本距离近
  3. normalize_embeddings=True 使向量归一化，后续用内积=余弦相似度
  4. 并行：利用GPU多流并行 + CPU多进程加速
  5. 断点续传：每批处理完立即写入，崩溃后从上次位置继续
输入：data/chunks/招股说明书_分块.jsonl
输出：data/chunks/招股说明书_带向量.jsonl

面试官可能问：
  Q: 为什么要并行？GPU本身不就是并行的吗？
  A: GPU并行指的是单次前向传播内的矩阵运算并行。但Python的
     sentence_transformers.encode()在batch之间是串行的——一个batch
     处理完才开始下一个。并行指的是多进程同时发送batch到GPU，
     减少GPU空闲时间（数据预处理、IO等）。
     RTX 4060有8GB显存，BGE-M3约占1.5GB，剩余空间可以同时跑2-3个batch。
"""

import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    get_kb_paths,
    CHUNKS_DIR,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    RUN_TIMESTAMP,
    setup_logger,
)

logger = setup_logger("generate_embeddings_parallel")


# =============================================================================
# 断点续传：检查已处理到哪一行
# =============================================================================
def get_checkpoint(output_path: Path) -> int:
    """
    作用：检查输出文件已处理到第几行，用于断点续传

    返回：
      已处理的chunk数量（0表示全新开始）

    原理：
      - 每处理完一批就追加写入JSONL
      - 重启时读取输出文件行数，跳过已处理的chunk
    """
    if not output_path.exists():
        return 0
    count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# =============================================================================
# 数据加载
# =============================================================================
def load_chunks(jsonl_path: str) -> list[dict]:
    """加载分块后的JSONL文件"""
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


# =============================================================================
# 单批编码函数（可被子进程调用）
# =============================================================================
def encode_batch(batch_data: list[tuple[int, str]], model_path: str) -> list[tuple[int, list[float]]]:
    """
    作用：编码一批文本，返回(idx, embedding)对

    参数：
      batch_data: [(原始索引, 文本), ...] 的列表
      model_path: 模型路径

    返回：
      [(原始索引, embedding列表), ...]

    为什么需要这个函数：
      - ProcessPoolExecutor要求函数可pickle
      - SentenceTransformer不能跨进程共享，每个子进程独立加载模型
      - 但GPU内存有限，不能同时加载太多模型实例

    面试官可能问：
      Q: 每个子进程都加载一次模型，不会很慢吗？
      A: 会。所以实际策略是：GPU只用单进程（避免显存竞争），
         CPU才用多进程。GPU的"并行"通过增大batch_size实现。
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_path)
    texts = [t for _, t in batch_data]
    indices = [i for i, _ in batch_data]

    embs = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )

    return [(idx, emb.tolist()) for idx, emb in zip(indices, embs)]


# =============================================================================
# 主向量化函数（支持断点续传）
# =============================================================================
def generate_embeddings(chunks: list[dict], output_path: Path) -> list[dict]:
    """
    作用：把chunks转成向量，支持断点续传

    设计思路：
      1. 检查已处理的checkpoint
      2. 跳过已处理的chunks
      3. 每批处理完立即追加写入（崩溃后不丢失）
      4. 全部完成后返回完整结果

    并行策略（RTX 4060 8GB）：
      - GPU模式：单进程，大batch_size（128）充分利用GPU并行
      - CPU模式：多进程（cpu_count//2），每个进程独立加载模型
      - 自动检测：如果GPU可用就用GPU，否则用CPU多进程

    面试官可能问：
      Q: 为什么GPU用单进程而不是多进程？
      A: PyTorch的CUDA操作默认是异步的，多个进程同时操作同一张GPU
         会导致显存竞争和CUDA错误。安全做法是单进程+大batch。
         如果想多进程GPU，需要用torch.multiprocessing.spawn，
         并且每个进程绑定不同的GPU（多卡场景）。

      Q: 断点续传的实现原理是什么？
      A: 每批encode完成后立即append写入JSONL文件。重启时先读取
         输出文件的行数作为checkpoint，跳过前N个chunk。这样即使
         中途崩溃，已完成的部分不会丢失。
    """
    if not chunks:
        logger.error("没有chunk数据")
        return []

    mp = str(EMBEDDING_MODEL_PATH)
    if not Path(mp).exists():
        logger.error(f"模型路径不存在: {mp}")
        return []

    # 检查断点
    checkpoint = get_checkpoint(output_path)
    if checkpoint > 0:
        logger.info(f"检测到断点：已处理 {checkpoint} 条，从第 {checkpoint + 1} 条继续")
        # 验证断点有效性
        if checkpoint > len(chunks):
            logger.warning(f"断点数({checkpoint})大于chunk数({len(chunks)})，重新开始")
            checkpoint = 0

    remaining = chunks[checkpoint:]
    if not remaining:
        logger.info("所有chunks已处理完成，无需重新编码")
        return load_existing_embeddings(output_path)

    logger.info(f"待处理: {len(remaining)}/{len(chunks)} 个chunks")

    # 检测GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else "无"
    except ImportError:
        has_gpu = False
        gpu_name = "无"

    logger.info(f"GPU: {gpu_name}, 可用: {has_gpu}")

    # -------------------------------------------------------------------------
    # GPU模式：单进程大batch
    # -------------------------------------------------------------------------
    if has_gpu:
        logger.info("GPU模式：单进程大batch编码")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(mp)
        logger.info(f"模型加载完成，输出维度: {EMBEDDING_DIM}")

        batch_size = 128  # RTX 4060 8GB足够
        total = len(remaining)
        processed = 0

        # 打开文件用于追加写入
        mode = "a" if checkpoint > 0 else "w"
        with open(output_path, mode, encoding="utf-8") as f:
            for i in range(0, total, batch_size):
                batch = remaining[i:i + batch_size]
                texts = [c["text"] for c in batch]

                try:
                    embs = model.encode(
                        texts,
                        show_progress_bar=False,
                        batch_size=len(texts),
                        normalize_embeddings=True,
                    )
                except Exception as e:
                    logger.error(f"第{i}-{i+len(batch)}条编码失败: {e}")
                    continue

                # 立即写入（断点续传）
                for c, v in zip(batch, embs):
                    record = {
                        "chunk_id": c["chunk_id"],
                        "page_start": c.get("page_start", 0) or 0,
                        "page_end": c.get("page_end", 0) or 0,
                        "section": c.get("section", ""),
                        "source_pdf": c.get("source_pdf", ""),
                        "type": c.get("type", "text"),
                        "table_index": c.get("table_index", 0),
                        "embedding": v.tolist(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1

                # 每批flush一次
                f.flush()
                logger.info(f"  已处理: {processed}/{total} ({processed/total*100:.1f}%)")

    # -------------------------------------------------------------------------
    # CPU模式：多进程并行
    # -------------------------------------------------------------------------
    else:
        logger.info("CPU模式：多进程并行编码")
        num_workers = max(1, cpu_count() // 2)
        logger.info(f"使用 {num_workers} 个进程")

        # 分批
        batch_size = 64
        batches = []
        for i in range(0, len(remaining), batch_size):
            batch = [(j, c["text"]) for j, c in enumerate(remaining[i:i + batch_size], i)]
            batches.append(batch)

        processed = 0
        mode = "a" if checkpoint > 0 else "w"

        with open(output_path, mode, encoding="utf-8") as f:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(encode_batch, batch, mp): idx
                    for idx, batch in enumerate(batches)
                }

                for future in as_completed(futures):
                    batch_idx = futures[future]
                    try:
                        results = future.result()
                        # 按原始索引排序
                        results.sort(key=lambda x: x[0])

                        for idx, emb in results:
                            c = remaining[idx]
                            record = {
                                "chunk_id": c["chunk_id"],
                                "page_start": c.get("page_start", 0) or 0,
                                "page_end": c.get("page_end", 0) or 0,
                                "section": c.get("section", ""),
                                "source_pdf": c.get("source_pdf", ""),
                                "type": c.get("type", "text"),
                                "table_index": c.get("table_index", 0),
                                "embedding": emb,
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            processed += 1

                        f.flush()
                        logger.info(f"  批次{batch_idx}完成, 已处理: {processed}/{len(remaining)}")

                    except Exception as e:
                        logger.error(f"  批次{batch_idx}失败: {e}")

    logger.info(f"向量化完成，共处理 {processed} 条")
    return load_existing_embeddings(output_path)


def load_existing_embeddings(output_path: Path) -> list[dict]:
    """加载已保存的向量文件"""
    if not output_path.exists():
        return []
    result = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return result


# =============================================================================
# 质量报告
# =============================================================================
def print_quality_report(chunks: list[dict]):
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


# =============================================================================
# 程序入口
# =============================================================================
def main(kb_name: str = None):
    logger.info("=" * 50)
    logger.info("  并行文本转向量 开始")
    logger.info("=" * 50)

    import os
    if kb_name is None:
        kb_name = os.environ.get("KB_NAME", "招股说明书1")
    kb_paths = get_kb_paths(kb_name)
    chunks_jsonl_path = kb_paths["chunks_jsonl_path"]
    embeddings_jsonl_path = Path(str(kb_paths["embeddings_jsonl_path"]))

    # 检查输入
    if not Path(str(chunks_jsonl_path)).exists():
        logger.error(f"输入文件不存在: {chunks_jsonl_path}")
        return

    # 备份（仅全新开始时备份）
    checkpoint = get_checkpoint(embeddings_jsonl_path)
    if checkpoint == 0 and embeddings_jsonl_path.exists():
        bn = f"{embeddings_jsonl_path.stem}_备份_{RUN_TIMESTAMP}{embeddings_jsonl_path.suffix}"
        shutil.copy2(embeddings_jsonl_path, embeddings_jsonl_path.parent / bn)
        logger.info(f"已备份旧文件 -> {embeddings_jsonl_path.parent / bn}")

    # 加载chunks
    chunks = load_chunks(str(chunks_jsonl_path))
    if not chunks:
        logger.error("未加载到数据")
        return
    logger.info(f"加载了 {len(chunks)} 个chunks")

    # 向量化
    start = time.time()
    vecs = generate_embeddings(chunks, embeddings_jsonl_path)
    elapsed = time.time() - start
    logger.info(f"向量化耗时: {elapsed:.1f}秒, 平均 {elapsed/len(chunks)*1000:.1f}ms/chunk")

    # 质量报告
    print_quality_report(vecs)
    logger.info("🎉 转向量完成")


if __name__ == "__main__":
    import os, sys
    kb = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KB_NAME", "招股说明书1")
    main(kb_name=kb)
