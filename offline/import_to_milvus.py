"""
import_to_milvus.py — 向量导入Milvus脚本
===========================================
作用：读取向量JSONL，补回文本，导入Milvus向量数据库
原理：
  1. Milvus是高性能分布式向量数据库，支持十亿级向量检索
  2. 使用gRPC协议通信，比HTTP更高效
  3. 向量文件已与文本解耦，导入时需要从chunk JSONL补回text
  4. 幂等设计：已有数据则跳过，避免重复导入
  5. REBUILD开关：True时删除旧集合重新创建
输入：data/chunks/招股说明书_带向量.jsonl + data/chunks/招股说明书_分块.jsonl
输出：Milvus集合 prospectus
"""

import json  # JSON序列化/反序列化
import sys  # 系统相关功能
from pathlib import Path  # 面向对象的文件路径处理

# pymilvus是Milvus的官方Python SDK
# 提供连接、集合管理、数据操作、检索等功能
from pymilvus import (
    connections,  # 连接管理
    utility,  # 工具函数（检查集合是否存在、删除集合等）
    FieldSchema,  # 字段定义
    CollectionSchema,  # 集合结构定义
    DataType,  # 数据类型枚举
    Collection,  # 集合操作类
)

# =============================================================================
# 模块路径设置
# =============================================================================
# 将项目根目录添加到Python模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 从项目根目录导入配置
from config import (
    EMBEDDINGS_JSONL_PATH,  # 向量结果输入路径
    CHUNKS_JSONL_PATH,  # 分块结果路径（用于补回文本）
    MILVUS_HOST,  # Milvus服务主机
    MILVUS_PORT,  # Milvus服务端口
    MILVUS_COLLECTION,  # Milvus集合名称
    MILVUS_BATCH_SIZE,  # 批量导入大小
    EMBEDDING_DIM,  # 向量维度（1024）
    REBUILD_COLLECTION,  # 是否重建集合
    RUN_TIMESTAMP,  # 运行时间戳
    setup_logger,  # 日志设置函数
)

# 创建日志记录器，名称为"import_to_milvus"
logger = setup_logger("import_to_milvus")


# =============================================================================
# 连接函数
# =============================================================================
def connect_milvus() -> bool:
    """
    作用：连接到本地Milvus服务
    
    原理：
      - pymilvus通过gRPC协议连接Milvus的19530端口
      - gRPC是高性能的远程过程调用协议，基于HTTP/2
      - 连接是全局的，后续操作无需重复连接
    
    返回：
      True: 连接成功
      False: 连接失败（记录错误日志）
    
    常见失败原因：
      - Milvus服务未启动
      - 主机/端口配置错误
      - 网络不通
    """
    logger.info(f"连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    try:
        # connections.connect() 建立全局连接
        # alias="default" 使用默认连接别名
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Milvus连接成功")
        return True
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return False


# =============================================================================
# 集合创建函数
# =============================================================================
def create_collection() -> Collection:
    """
    作用：在Milvus中创建prospectus集合，定义字段结构
    
    原理：
      - Milvus集合类似数据库的表，需要提前定义字段结构
      - 字段定义后不可修改，需要规划好字段类型和长度
      - 向量字段需要指定维度（dim），且所有向量必须同维度
    
    字段设计：
      - id: 自增主键，Milvus自动生成
      - chunk_id: 业务主键，用于关联文本文件
      - text: 文本内容，VARCHAR最大10000字符
      - page_start/page_end: 页码范围，INT64类型
      - section: 所属章节，VARCHAR最大100字符
      - source_pdf: 来源PDF路径，VARCHAR最大500字符
      - content_type: 内容类型（text/table），VARCHAR最大20字符
      - table_index: 表格编号，INT64类型
      - embedding: 向量字段，FLOAT_VECTOR类型，1024维
    
    返回：
      Collection对象，可用于后续数据操作
    """
    # 定义字段列表
    fields = [
        # -----------------------------------------------------------------
        # id: 自增主键
        # -----------------------------------------------------------------
        # is_primary=True: 主键字段
        # auto_id=True: Milvus自动生成，无需手动指定
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        
        # -----------------------------------------------------------------
        # chunk_id: 业务主键
        # -----------------------------------------------------------------
        # VARCHAR: 变长字符串
        # max_length=50: 最大50字符，chunk_id格式如"chunk_0001"
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=50),
        
        # -----------------------------------------------------------------
        # text: 文本内容
        # -----------------------------------------------------------------
        # max_length=10000: 最大10000字符，防止超长chunk
        # 如果chunk超过此长度，会被截断
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        
        # -----------------------------------------------------------------
        # page_start/page_end: 页码范围
        # -----------------------------------------------------------------
        # INT64: 64位整数，足够存储页码
        FieldSchema(name="page_start", dtype=DataType.INT64),
        FieldSchema(name="page_end", dtype=DataType.INT64),
        
        # -----------------------------------------------------------------
        # section: 所属章节
        # -----------------------------------------------------------------
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
        
        # -----------------------------------------------------------------
        # source_pdf: 来源PDF路径
        # -----------------------------------------------------------------
        FieldSchema(name="source_pdf", dtype=DataType.VARCHAR, max_length=500),
        
        # -----------------------------------------------------------------
        # content_type: 内容类型（新增字段，支持表格）
        # -----------------------------------------------------------------
        # "text" 或 "table"，用于区分普通文本和表格
        FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=20),
        
        # -----------------------------------------------------------------
        # table_index: 表格编号（新增字段）
        # -----------------------------------------------------------------
        # 表格类型时有效，标记是第几个表格
        FieldSchema(name="table_index", dtype=DataType.INT64),
        
        # -----------------------------------------------------------------
        # embedding: 向量字段
        # -----------------------------------------------------------------
        # FLOAT_VECTOR: 浮点向量
        # dim=EMBEDDING_DIM: 向量维度，必须与BGE-M3输出一致（1024）
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    
    # 创建集合结构
    schema = CollectionSchema(fields, description="招股说明书问答系统向量库（支持表格）")
    
    # 创建集合
    c = Collection(name=MILVUS_COLLECTION, schema=schema)
    logger.info(f"集合 '{MILVUS_COLLECTION}' 创建成功")
    return c


# =============================================================================
# 索引创建函数
# =============================================================================
def create_index(collection):
    """
    作用：为向量字段创建索引，加速检索
    
    原理：
      - 没有索引时，Milvus需要全量扫描（暴力搜索），速度慢
      - IVF_FLAT是倒排文件索引，适合中小规模数据（百万级）
      - nlist=128: 将向量空间划分为128个区域（聚类中心）
      - 检索时先定位到最近的几个区域，再精确计算，大幅提升速度
    
    参数：
      collection: Collection对象
    
    索引类型对比：
      - IVF_FLAT: 速度快，召回率高，适合百万级数据
      - IVF_SQ8: 压缩存储，省空间，适合亿级数据
      - HNSW: 图索引，速度最快，适合高并发场景
    
    距离度量：
      - IP（内积）: 向量已归一化时，内积=余弦相似度
      - L2（欧氏距离）: 适合未归一化的向量
    """
    # 索引参数
    params = {
        "metric_type": "IP",  # 内积距离（向量已归一化，内积=余弦相似度）
        "index_type": "IVF_FLAT",  # 倒排文件索引
        "params": {"nlist": 128},  # 聚类中心数
    }
    try:
        collection.create_index(field_name="embedding", index_params=params)
        logger.info("向量索引创建成功")
    except Exception as e:
        # 索引可能已存在（如集合已存在时）
        logger.warning(f"索引可能已存在: {e}")


# =============================================================================
# 数据检查函数
# =============================================================================
def check_existing_data(collection) -> int:
    """
    作用：检查集合中已有数据量
    
    原理：
      - 幂等设计：已有数据则跳过导入，避免重复
      - collection.load() 将集合加载到内存
      - collection.num_entities 返回数据条数
    
    参数：
      collection: Collection对象
    
    返回：
      数据条数（0表示空集合）
    """
    try:
        collection.load()  # 加载集合到内存
        return collection.num_entities  # 获取数据条数
    except Exception:
        return 0  # 加载失败，视为空集合


# =============================================================================
# 集合存在检查函数
# =============================================================================
def check_collection_exists(name: str) -> bool:
    """
    作用：检查集合是否已存在
    
    原理：
      - utility.has_collection() 检查指定名称的集合是否存在
      - 用于决定是创建新集合还是使用已有集合
    
    参数：
      name: 集合名称
    
    返回：
      True: 集合存在
      False: 集合不存在
    """
    return utility.has_collection(name)


# =============================================================================
# 文本映射加载函数
# =============================================================================
def load_text_map() -> dict:
    """
    作用：从chunk JSONL加载文本映射（chunk_id → text）
    
    原理：
      - 向量文件已与文本解耦（不存text），导入Milvus时需要补回
      - 建立chunk_id到text的字典映射，导入时通过chunk_id查找
      - 解耦设计的好处：向量文件小，文本可独立更新
    
    返回：
      {chunk_id: text, ...} 字典
    
    异常处理：
      - 文件不存在：记录警告，返回空字典
      - 单条解析失败：跳过该条，继续处理
    """
    m = {}  # 映射字典
    p = Path(str(CHUNKS_JSONL_PATH))
    
    if not p.exists():
        logger.warning(f"chunk文件不存在: {CHUNKS_JSONL_PATH}")
        return m
    
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                d = json.loads(line)  # 解析JSON
                m[d["chunk_id"]] = d["text"]  # 建立映射
            except (json.JSONDecodeError, KeyError):
                continue  # 解析失败或缺少字段，跳过
    
    logger.info(f"已加载{len(m)}条文本映射")
    return m


# =============================================================================
# 数据导入函数
# =============================================================================
def load_data(collection):
    """
    作用：从向量JSONL读取数据，补回文本，批量导入Milvus
    
    原理：
      1. 幂等检查：已有数据则跳过
      2. 加载文本映射：从chunk JSONL补回text
      3. 解析向量JSONL：逐行读取，组装成列格式
      4. 维度校验：确保所有向量维度一致
      5. 批量导入：每批MILVUS_BATCH_SIZE条，减少网络往返
    
    参数：
      collection: Collection对象
    
    数据格式转换：
      JSONL文件是行式存储（每行一条记录）
      Milvus需要列式存储（每个字段一个列表）
      例如：
        行式：[{"chunk_id":"a", "text":"xxx"}, {"chunk_id":"b", "text":"yyy"}]
        列式：chunk_ids=["a","b"], texts=["xxx","yyy"]
    """
    # -------------------------------------------------------------------------
    # 步骤1：幂等检查
    # -------------------------------------------------------------------------
    existing = check_existing_data(collection)
    if existing > 0:
        # 已有数据，跳过导入
        logger.info(f"集合已有{existing}条数据，跳过导入（如需重建设REBUILD_COLLECTION=True）")
        return
    
    # -------------------------------------------------------------------------
    # 步骤2：检查向量文件
    # -------------------------------------------------------------------------
    ip = Path(str(EMBEDDINGS_JSONL_PATH))
    if not ip.exists():
        logger.error(f"文件不存在: {EMBEDDINGS_JSONL_PATH}")
        return
    
    logger.info(f"加载向量数据: {EMBEDDINGS_JSONL_PATH}")
    
    # -------------------------------------------------------------------------
    # 步骤3：加载文本映射
    # -------------------------------------------------------------------------
    text_map = load_text_map()
    
    # -------------------------------------------------------------------------
    # 步骤4：解析向量JSONL，组装列式数据
    # -------------------------------------------------------------------------
    # 初始化9个空列表，分别对应9个字段
    # cids: chunk_id列表
    # texts: text列表（从text_map补回）
    # pss: page_start列表
    # pes: page_end列表
    # secs: section列表
    # sps: source_pdf列表
    # ctypes: content_type列表
    # tidxs: table_index列表
    # embs: embedding列表
    cids, texts, pss, pes, secs, sps, ctypes, tidxs, embs = ([] for _ in range(9))
    
    with open(ip, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                d = json.loads(line)  # 解析JSON
                cid = d["chunk_id"]  # chunk标识
                cids.append(cid)
                texts.append(text_map.get(cid, ""))  # 从映射补回text，找不到则为空
                pss.append(d.get("page_start", 0))  # 起始页码
                pes.append(d.get("page_end", 0))  # 结束页码
                secs.append(d.get("section") or "")  # 所属章节
                sps.append(d.get("source_pdf", ""))  # 来源PDF
                # 表格相关字段
                ctypes.append(d.get("type", "text"))  # 内容类型
                tidxs.append(d.get("table_index", 0))  # 表格编号
                embs.append(d["embedding"])  # 向量数据
            except (json.JSONDecodeError, KeyError) as e:
                # 解析失败或缺少关键字段，记录警告
                logger.warning(f"第{ln}行解析失败: {e}")
    
    total = len(cids)  # 总数据条数
    logger.info(f"共加载{total}条")
    if not total:
        return  # 没有数据，提前退出
    
    # -------------------------------------------------------------------------
    # 步骤5：向量维度校验
    # -------------------------------------------------------------------------
    # 所有向量维度必须一致，否则Milvus会报错
    first_dim = len(embs[0])  # 第一条向量的维度
    for i, e in enumerate(embs):
        if len(e) != first_dim:
            logger.error(f"第{i}条向量维度不一致: 期望{first_dim}, 实际{len(e)}")
            return  # 维度不一致，终止导入
    
    logger.info(f"向量维度校验通过: {first_dim}维")
    
    # -------------------------------------------------------------------------
    # 步骤6：批量导入
    # -------------------------------------------------------------------------
    # range(0, total, MILVUS_BATCH_SIZE): 从0开始，每次步进100
    # 例如：total=250, batch_size=100
    #   第1批：0-100
    #   第2批：100-200
    #   第3批：200-250
    for i in range(0, total, MILVUS_BATCH_SIZE):
        end = min(i + MILVUS_BATCH_SIZE, total)  # 计算批次结束位置
        
        # 组装批次数据：每个字段取切片 [i:end]
        batch = [
            cids[i:end],      # chunk_id列表
            texts[i:end],     # text列表
            pss[i:end],       # page_start列表
            pes[i:end],       # page_end列表
            secs[i:end],      # section列表
            sps[i:end],       # source_pdf列表
            ctypes[i:end],    # content_type列表
            tidxs[i:end],     # table_index列表
            embs[i:end],      # embedding列表
        ]
        
        try:
            collection.insert(batch)  # 批量插入
            logger.info(f"已导入{end}/{total}条")
        except Exception as e:
            logger.error(f"第{i}-{end}条导入失败: {e}")
    
    # -------------------------------------------------------------------------
    # 步骤7：刷新数据
    # -------------------------------------------------------------------------
    # flush() 将内存中的数据持久化到存储
    # 确保数据立即写入，后续查询可见
    collection.flush()
    logger.info(f"导入完成，集合当前数据量: {collection.num_entities}")


# =============================================================================
# 质量报告函数
# =============================================================================
def print_quality_report(collection):
    """
    作用：输出导入质量报告
    
    原理：
      - 加载集合并统计数据量
      - 显示集合名、数据量、向量维度
      - 帮助确认导入是否成功
    
    参数：
      collection: Collection对象
    """
    try:
        collection.load()  # 加载集合到内存
        
        logger.info("=" * 50)
        logger.info("  导入质量报告")
        logger.info("=" * 50)
        logger.info(f"集合: {MILVUS_COLLECTION}")
        logger.info(f"数据量: {collection.num_entities}条")
        logger.info(f"向量维度: {EMBEDDING_DIM}")
    except Exception as e:
        logger.warning(f"质量报告获取失败: {e}")


# =============================================================================
# 程序入口
# =============================================================================
def main():
    """
    作用：程序入口，串联导入流程
    
    执行流程：
      1. 连接Milvus
      2. 根据REBUILD_COLLECTION决定：
         - True: 删除旧集合，创建新集合，创建索引
         - False: 检查集合是否存在
           - 存在：直接使用
           - 不存在：创建新集合，创建索引
      3. 导入数据
      4. 输出质量报告
    
    REBUILD_COLLECTION使用场景：
      - True: 首次导入，或需要清空旧数据重新导入
      - False: 增量导入，或避免误删数据
    """
    logger.info("=" * 50)
    logger.info("  导入Milvus 开始")
    logger.info("=" * 50)
    
    # 步骤1：连接Milvus
    if not connect_milvus():
        return  # 连接失败，提前退出
    
    # 步骤2：处理集合（重建或复用）
    if REBUILD_COLLECTION:
        # -----------------------------------------------------------------
        # 重建模式：删除旧集合，创建新集合
        # -----------------------------------------------------------------
        if utility.has_collection(MILVUS_COLLECTION):
            logger.info("REBUILD_COLLECTION=True，删除旧集合...")
            utility.drop_collection(MILVUS_COLLECTION)  # 删除旧集合
        c = create_collection()  # 创建新集合
        create_index(c)  # 创建索引
    elif check_collection_exists(MILVUS_COLLECTION):
        # -----------------------------------------------------------------
        # 复用模式：集合已存在，直接使用
        # -----------------------------------------------------------------
        logger.info(f"集合已存在，直接使用")
        c = Collection(MILVUS_COLLECTION)
    else:
        # -----------------------------------------------------------------
        # 新建模式：集合不存在，创建新集合
        # -----------------------------------------------------------------
        c = create_collection()
        create_index(c)
    
    # 步骤3：导入数据
    load_data(c)
    
    # 步骤4：输出质量报告
    print_quality_report(c)
    
    logger.info("🎉 导入完成")


# =============================================================================
# 脚本执行入口
# =============================================================================
if __name__ == "__main__":
    main()
