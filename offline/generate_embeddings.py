"""
generate_embeddings.py — 文本转向量脚本
=========================================
作用：加载BGE-M3模型，把chunks转成1024维向量，文本与向量解耦存储
原理：
  1. BGE-M3是北京智源研究院开源的多语言Embedding模型
  2. 将文本映射到高维向量空间，语义相似的文本距离近
  3. normalize_embeddings=True 使向量归一化，后续用内积=余弦相似度
  4. 文本和向量分开存储：JSONL存文本，向量文件存chunk_id+向量
输入：data/chunks/招股说明书_分块.jsonl
输出：data/chunks/招股说明书_带向量.jsonl
"""

import json  # JSON序列化/反序列化
import shutil  # 文件操作工具，用于备份
import sys  # 系统相关功能
from pathlib import Path  # 面向对象的文件路径处理

# sentence-transformers是Embedding模型的Python库
# 封装了HuggingFace Transformers，提供简洁的API
from sentence_transformers import SentenceTransformer

# =============================================================================
# 模块路径设置
# =============================================================================
# 将项目根目录添加到Python模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 从项目根目录导入配置
from config import (
    get_kb_paths,  # v2.0: 获取指定KB的文件路径
    CHUNKS_DIR,  # chunks目录
    EMBEDDING_MODEL_PATH,  # BGE-M3模型本地路径
    EMBEDDING_BATCH_SIZE,  # 批量编码大小
    EMBEDDING_DIM,  # 向量维度（1024）
    RUN_TIMESTAMP,  # 运行时间戳
    setup_logger,  # 日志设置函数
)

# 创建日志记录器，名称为"generate_embeddings"
logger = setup_logger("generate_embeddings")


# =============================================================================
# 文件检查函数
# =============================================================================
def check_file_exists(file_path: str) -> bool:
    """
    作用：检查输入文件是否存在
    
    原理：向量化依赖于分块结果，如果输入文件不存在则无法继续
    
    参数：
      file_path: 文件路径字符串
    
    返回：
      True: 文件存在
      False: 文件不存在（同时记录错误日志）
    """
    p = Path(file_path)
    if not p.exists():
        logger.error(f"文件不存在: {file_path}")
        return False
    return True


# =============================================================================
# 备份函数
# =============================================================================
def backup_previous_output(output_path: str):
    """
    作用：备份旧的向量文件，用于回滚
    
    原理：向量化可能覆盖旧文件，备份后如果新结果有问题可以恢复
    
    参数：
      output_path: 输出文件路径
    
    备份命名规则：
      原文件名_备份_时间戳.后缀
    """
    p = Path(output_path)
    if p.exists():
        bn = f"{p.stem}_备份_{RUN_TIMESTAMP}{p.suffix}"
        shutil.copy2(p, p.parent / bn)
        logger.info(f"已备份旧文件 -> {p.parent / bn}")


# =============================================================================
# 数据加载函数
# =============================================================================
def load_chunks(jsonl_path: str) -> list[dict]:
    """
    作用：加载分块后的JSONL文件
    
    原理：
      - JSONL（JSON Lines）格式：每行一条JSON记录
      - 适合流式处理大文件，无需一次性加载整个文件到内存
      - 逐行解析，单条失败不影响其他记录
    
    参数：
      jsonl_path: JSONL文件路径
    
    返回：
      chunk字典列表
    
    异常处理：
      - 单条JSON解析失败：记录警告，跳过该条，继续处理
      - 空行：自动跳过
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                chunks.append(json.loads(line))  # 解析JSON
            except json.JSONDecodeError as e:
                # 单条解析失败，记录警告但不中断
                logger.warning(f"第{ln}行解析失败: {e}")
    return chunks


# =============================================================================
# 向量化函数
# =============================================================================
def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    作用：加载BGE-M3模型，把所有chunks的文字转成向量
    
    原理：
      - BGE-M3是一个多语言嵌入模型，输出1024维向量
      - 语义相似的文本在高维空间中距离近（余弦相似度高）
      - normalize_embeddings=True 使向量归一化，后续用内积=余弦相似度
      - 批量处理（batch）提高效率，充分利用GPU并行计算能力
    
    参数：
      chunks: 分块后的chunk列表，每个chunk必须有"text"字段
    
    返回：
      带向量的chunk元数据列表（不含text字段，实现解耦）
    
    执行流程：
      1. 检查模型路径是否存在
      2. 加载BGE-M3模型（自动使用GPU如果可用）
      3. 提取所有chunk的text字段
      4. 批量encode（显示进度条）
      5. 校验向量数量和维度
      6. 解耦存储：只保留chunk_id+元数据+向量
    
    解耦设计：
      - 向量文件不存储text内容，只存储chunk_id
      - 检索时用chunk_id关联到文本文件
      - 好处：向量文件小，文本可独立更新

    面试官可能问：
      Q: 为什么用SentenceTransformer加载BGE-M3，不是直接用transformers？
      A: SentenceTransformer封装了pooling和normalization，开箱即用。
         如果用transformers的AutoModel，需要手动实现pooling策略
         （mean pooling / cls pooling）和归一化，容易出错。
         SentenceTransformer是社区标准做法。

      Q: normalize_embeddings=True有什么作用？
      A: 使输出向量的L2范数为1（向量长度=1）。归一化后，
         内积（IP）= 余弦相似度。Milvus检索时用IP度量，
         等价于用余弦相似度排序。如果不归一化，长向量的内积
         天然大于短向量，会影响检索公平性。

      Q: 为什么要把text从向量文件中解耦出去？
      A: 一个1024维的float向量约4KB（1024 * 4字节），
         一条text约500-800字符（UTF-8约1-2KB）。
         如果不解耦，向量文件大小翻倍。更重要的是：
         检索时只用到embedding字段，text只在返回结果时用，
         分开后检索效率更高。

      Q: 模型加载失败后为什么返回空列表而不是抛异常？
      A: 离线管道的设计是"检查后中止"而非"运行时崩溃"。
         generate_embeddings返回空列表后，调用方main()检查到
         列表为空就记录错误并退出。用户知道"模型加载失败"，
         不需要去看堆栈跟踪。

      Q: 为什么用numpy的v.tolist()转为Python列表？
      A: JSON不支持numpy数组序列化。v.tolist()把numpy.ndarray
         转为嵌套的Python list，json.dump就能正常写入。
         如果数据量大，改为np.save用二进制格式更快
         （当前数据量小，JSONL足够）。
    """
    if not chunks:
        logger.error("没有chunk数据")
        return []
    
    # 检查模型路径
    mp = str(EMBEDDING_MODEL_PATH)
    if not Path(mp).exists():
        logger.error(f"模型路径不存在: {mp}")
        return []
    
    # -------------------------------------------------------------------------
    # 步骤1：加载模型
    # -------------------------------------------------------------------------
    logger.info("加载BGE-M3模型...")
    try:
        # SentenceTransformer自动处理模型加载、tokenizer、设备选择（CPU/GPU）
        model = SentenceTransformer(mp)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return []
    logger.info(f"模型加载完成，输出维度: {EMBEDDING_DIM}")
    
    # -------------------------------------------------------------------------
    # 步骤2：提取文本
    # -------------------------------------------------------------------------
    texts = [c["text"] for c in chunks]  # 提取所有chunk的文本内容
    logger.info(f"共{len(texts)}个chunks，转向量...")
    
    # -------------------------------------------------------------------------
    # 步骤3：批量编码
    # -------------------------------------------------------------------------
    try:
        embs = model.encode(
            texts,  # 文本列表
            show_progress_bar=True,  # 显示进度条（Tqdm）
            batch_size=EMBEDDING_BATCH_SIZE,  # 每批处理32条
            normalize_embeddings=True,  # 归一化：向量长度=1，内积=余弦相似度
        )
    except Exception as e:
        logger.error(f"向量化失败: {e}")
        return []
    
    logger.info(f"向量化完成，维度: {embs.shape[1]}")
    
    # -------------------------------------------------------------------------
    # 步骤4：校验
    # -------------------------------------------------------------------------
    # 校验1：向量数必须等于chunk数
    if len(embs) != len(texts):
        logger.error(f"向量数不匹配: chunks {len(texts)}, embeddings {len(embs)}")
        return []
    
    # 校验2：向量维度必须匹配配置
    if embs.shape[1] != EMBEDDING_DIM:
        logger.warning(f"维度不匹配: 期望{EMBEDDING_DIM}, 实际{embs.shape[1]}")
    
    # -------------------------------------------------------------------------
    # 步骤5：解耦存储（只存元数据+向量，不存text）
    # -------------------------------------------------------------------------
    result = []
    ok = 0  # 成功计数
    
    for c, v in zip(chunks, embs):
        try:
            result.append({
                "chunk_id": c["chunk_id"],  # 唯一标识，用于关联文本
                "page_start": c.get("page_start", 0) or 0,  # 起始页码
                "page_end": c.get("page_end", 0) or 0,  # 结束页码
                "section": c.get("section", ""),  # 所属章节
                "source_pdf": c.get("source_pdf", ""),  # 来源PDF
                "type": c.get("type", "text"),  # 内容类型：text/table
                "table_index": c.get("table_index", 0),  # 表格编号
                "embedding": v.tolist(),  # numpy数组转为Python列表（JSON可序列化）
            })
            ok += 1
        except Exception as e:
            logger.warning(f"chunk {c.get('chunk_id','?')} 处理失败: {e}")
    
    logger.info(f"转向量完成: 成功{ok}条")
    return result


# =============================================================================
# 保存函数
# =============================================================================
def save_embeddings(chunks: list[dict], output_path: Path):
    """
    作用：保存带向量的chunks为JSONL文件
    
    原理：
      - JSONL格式适合存储大量记录
      - 每行一条记录，便于流式读取
      - 使用ensure_ascii=False保留中文字符
    
    参数：
      chunks: 带向量的chunk列表
    """
    op = Path(str(output_path))
    op.parent.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
    
    with open(op, "w", encoding="utf-8") as f:
        for c in chunks:
            # json.dumps：将字典转为JSON字符串
            # ensure_ascii=False：保留中文字符，不转义为\uXXXX
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    logger.info(f"已保存: {op}")


# =============================================================================
# 质量报告函数
# =============================================================================
def print_quality_report(chunks: list[dict]):
    """
    作用：输出向量化质量报告
    
    原理：
      - 统计总chunk数、含向量的比例
      - 检查向量维度是否一致
      - 发现异常（如部分chunk未向量化）
    
    参数：
      chunks: 带向量的chunk列表
    """
    if not chunks:
        return
    
    total = len(chunks)  # 总chunk数
    with_emb = sum(1 for c in chunks if "embedding" in c)  # 含向量的chunk数
    dims = set(len(c["embedding"]) for c in chunks if "embedding" in c)  # 维度集合
    
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
    """
    作用：程序入口，加载chunks→转向量→质量报告→保存

    参数：
      kb_name: 知识库名称（从环境变量 KB_NAME 获取）
    
    执行流程：
      1. 检查输入文件
      2. 备份旧输出
      3. 加载分块结果
      4. 批量向量化
      5. 输出质量报告
      6. 保存向量文件
    """
    logger.info("=" * 50)
    logger.info("  文本转向量 开始")
    logger.info("=" * 50)

    # v2.0: 确定知识库名称
    import os
    if kb_name is None:
        kb_name = os.environ.get("KB_NAME", "招股说明书1")
    kb_paths = get_kb_paths(kb_name)
    chunks_jsonl_path = kb_paths["chunks_jsonl_path"]
    embeddings_jsonl_path = kb_paths["embeddings_jsonl_path"]

    # 步骤1：检查输入文件
    if not check_file_exists(str(chunks_jsonl_path)):
        return

    # 步骤2：备份旧输出
    backup_previous_output(str(embeddings_jsonl_path))

    # 步骤3：加载分块结果
    chunks = load_chunks(str(chunks_jsonl_path))
    if not chunks:
        logger.error("未加载到数据")
        return
    logger.info(f"加载了{len(chunks)}个chunks")
    
    # 步骤4：向量化
    vecs = generate_embeddings(chunks)
    if not vecs:
        logger.error("向量化失败")
        return
    
    # 步骤5：输出质量报告
    print_quality_report(vecs)
    
    # 步骤6：保存
    save_embeddings(vecs, embeddings_jsonl_path)
    logger.info("🎉 转向量完成")


# =============================================================================
# 脚本执行入口
# =============================================================================
if __name__ == "__main__":
    import os, sys
    kb = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KB_NAME", "招股说明书1")
    main(kb_name=kb)
