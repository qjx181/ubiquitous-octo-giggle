"""Embedding服务：文本转向量（BGE-M3, 1024维），支持单条/批量编码。"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from ..config import settings

# 创建一个日志记录器，用于输出运行时信息（比如模型加载成功/失败、编码异常等）
# __name__ 是当前模块的名称，方便在日志中定位是哪段代码输出的
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Embedding服务，封装BGE-M3模型，提供文本转向量功能。

    返回类型：
      encode(text: str) → List[float] (1024维归一化向量)
      encode_batch(texts: List[str]) → List[List[float]]

    边界情况：
      - 模型加载失败 → 抛RuntimeError，让上层（probe）捕获后标记为不可用
      - encode空字符串 → BGE-M3也能正常编码（返回一个向量），不会报错
      - encode超长文本 → BGE-M3最大输入512 token，超长会自动截断
      - batch_size太大 → GPU显存溢出（OOM），降低batch_size重试

    面试官可能问：
      Q: 为什么要单独封装EmbeddingService，不用SentenceTransformer直接调用？
      A: 封装的好处：(1) 单例模式确保模型只加载一次；(2) 统一encode接口，
         调用方不需要关心模型细节；(3) 维度校验在加载时完成，运行时不必重复检查。

      Q: 为什么模型加载失败后抛异常而不是返回None？
      A: Embedding是RAG流程的基础（没有向量根本无法检索），
         如果Embedding失败，整个系统无法工作。所以必须抛异常让上层知道
         （而不是静默降级）。这和Reranker不同——Reranker是锦上添花，
         Embedding是雪中送炭。

      Q: encode方法为什么用normalize_embeddings=True？
      A: 归一化后内积=余弦相似度，Milvus用IP（内积）索引时等价于
         按余弦相似度排序。如果不归一化，长向量的内积天然大于短向量，
         检索结果会偏向文本长的文档，这是不公平的。

      Q: encode_batch的batch_size默认32，什么情况下需要调整？
      A: GPU显存决定batch_size上限。RTX 4060 8GB显存，batch_size=32
         是安全值。如果显存更大（如A100 80GB），可以提高batch_size
         到128-256以加快处理速度。如果显存不足（OOM），降低到16或8。
    """

    # 【作用】构造函数，创建 EmbeddingService 实例时自动调用
    # 【逻辑】先把 self.model 设为 None 占位，再调用 _load_model() 真正加载模型
    def __init__(self):
        """初始化：加载BGE-M3模型。"""
        # 【作用】将模型暂时设为 None，防止其他方法在模型未加载时误用
        # 【逻辑】_load_model() 方法里会重新给 self.model 赋值，这里只是占个坑
        self.model = None  # 先占位，_load_model()里真正加载
        # 【作用】调用内部方法，真正去加载 BGE-M3 模型到内存中
        # 【逻辑】这一步可能会比较慢（模型文件可能几百 MB），但只执行一次
        self._load_model()

    # 【作用】内部方法（以下划线开头表示"私有的，外部不要直接调用"），负责加载 BGE-M3 模型
    # 【逻辑】从配置中读取模型路径 → 判断用 CPU 还是 GPU → 加载模型 → 验证输出维度
    def _load_model(self):
        """加载BGE-M3模型，验证输出维度是否与配置一致。"""
        # 【作用】捕获整个加载过程中可能出现的任何异常，防止程序崩溃
        # 【逻辑】如果 try 块里任何一行出错，立刻跳到 except 块记录日志并抛出自定义异常
        try:
            # 【作用】从配置对象 settings 中读取模型文件所在路径
            # 【逻辑】settings.EMBEDDING_MODEL_PATH 在 config 模块中定义，通常是本地文件夹路径
            model_path = settings.EMBEDDING_MODEL_PATH
            # 【作用】输出一条 INFO 级别的日志，告诉运维人员当前正在加载哪个路径下的模型
            # 【逻辑】这样如果加载卡住了，看日志就能知道是哪个路径有问题
            logger.info(f"加载Embedding模型: {model_path}")

            # 加载模型
            # 原理：SentenceTransformer会自动从本地路径加载
            # 明确指定 device='cuda' 使用 GPU（如果有），加速编码
            # 【作用】导入 PyTorch 库，用于检测当前机器是否有可用的 GPU
            # 【逻辑】把 import 写在函数内部而不是文件顶部，是一种"懒加载"技巧——只有真正需要时才导入
            import torch
            # 【作用】判断 GPU 是否可用，决定模型跑在 GPU 还是 CPU 上
            # 【逻辑】torch.cuda.is_available() 返回 True 说明有 NVIDIA GPU 且驱动正常
            #        有 GPU 就用 'cuda'，没有就用 'cpu'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # 【作用】用 SentenceTransformer 库从本地路径加载 BGE-M3 模型
            # 【逻辑】model_path 指向下载好的模型文件夹，device 决定计算设备
            #        SentenceTransformer 会自动读取模型配置和权重文件
            self.model = SentenceTransformer(model_path, device=device)
            # 【作用】记录一条日志，告诉运维人员模型已经加载完毕，以及用的是 GPU 还是 CPU
            # 【逻辑】方便排查——如果模型加载慢或报错，这条日志能确认加载流程走到了哪一步
            logger.info(f"Embedding模型加载完成，使用设备: {device}")

            # 验证维度
            # 作用：确保模型输出维度（1024）和config里配置的一致
            # 如果维度不一致，Milvus检索时会报错（要求向量维度精确匹配）
            # 【作用】用一条测试文本 "测试" 跑一遍编码，拿到输出向量
            # 【逻辑】encode() 返回的是一个 numpy 数组，len() 可以拿到它的维度大小
            test_vector = self.model.encode("测试")
            # 【作用】检查实际输出维度是否和配置文件中声明的维度一致
            # 【逻辑】EMBEDDING_DIM 在 config 里定义（通常是1024）
            #        如果不一致，说明模型文件可能下错了或配置写错了
            if len(test_vector) != settings.EMBEDDING_DIM:
                # 【作用】如果维度不匹配，只是警告（warning）而不是报错（error）
                # 【逻辑】程序还能继续跑，但后续往 Milvus 存向量时可能会出问题
                logger.warning(
                    # 【作用】在警告信息里同时告诉用户期望值和实际值，方便定位问题
                    f"模型输出维度不匹配: "
                    f"期望{settings.EMBEDDING_DIM}, 实际{len(test_vector)}"
                )

        # 【作用】捕获 try 块中抛出的任何异常，统一处理
        # 【逻辑】不管是模型文件不存在、路径写错了、还是 GPU 内存不够，都走到这里
        except Exception as e:
            # 【作用】记录一条 ERROR 级别的日志，包含具体的异常信息
            # 【逻辑】这样运维人员能在日志文件中看到详细的错误原因
            logger.error(f"模型加载失败: {e}")
            # 【作用】抛出一个 RuntimeError 异常，让上层调用者知道模型加载失败了
            # 【逻辑】如果模型没加载成功，后续所有编码操作都会失败，不如及早报错
            raise RuntimeError(f"Embedding模型加载失败: {e}")

    # 【作用】对外提供的接口方法，把一段文本转换成向量（embedding vector）
    # 【参数】text: str —— 要编码的文本，比如"今天天气真好"
    # 【返回值】List[float] —— 一个包含浮点数的列表，长度固定为 1024
    def encode(self, text: str) -> List[float]:
        """将单段文本编码为1024维归一化向量（内积=余弦相似度）。"""
        # 【作用】捕获编码过程中可能出现的任何异常
        # 【逻辑】模型加载成功后 encode() 一般不会出错，但以防万一（比如传了空字符串）
        try:
            # encode() 返回numpy数组
            # normalize_embeddings=True → 向量归一化，内积=余弦相似度
            # 【作用】调用 SentenceTransformer 的 encode 方法，将文本转成向量
            # 【逻辑】normalize_embeddings=True 会对输出向量做 L2 归一化
            #        归一化之后，两个向量的点积（内积）就等于余弦相似度
            #        这在向量检索（如 Milvus）中非常有用
            vector = self.model.encode(
                text,
                normalize_embeddings=True
            )
            # tolist() 把numpy数组转成Python列表，方便存JSON
            # 【作用】将 numpy 数组转换成 Python 原生的列表
            # 【逻辑】因为后面要存到数据库或序列化成 JSON，numpy 类型不能被 json.dumps 直接处理
            return vector.tolist()

        # 【作用】捕获 encode 过程中抛出的任何异常
        except Exception as e:
            # 【作用】记录错误日志，包含具体的异常信息
            logger.error(f"文本编码失败: {e}")
            # 【作用】抛出一个 RuntimeError，让调用方知道编码失败了
            raise RuntimeError(f"文本编码失败: {e}")

    # 【作用】批量编码接口，一次处理多条文本，比逐条调用 encode() 更快
    # 【参数】texts: List[str] —— 一个字符串列表，比如 ["第一段", "第二段", "第三段"]
    # 【返回值】List[List[float]] —— 每个文本对应一个向量列表，外层列表长度 = 文本条数
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码多段文本（利用GPU并行，由EMBEDDING_BATCH_SIZE控制批大小）。"""
        # 【作用】捕获批量编码过程中可能出现的任何异常
        try:
            # batch_size参数控制一次处理多少条
            # 太小利用不了GPU并行，太大可能显存溢出
            # 【作用】调用 SentenceTransformer 的 encode 方法，一次性编码多条文本
            # 【逻辑】传入列表而不是单条文本，模型内部会并行计算（GPU 上并行尤其快）
            #        batch_size 控制每批处理多少条，默认值通常在 config 里配为 32
            #        show_progress_bar=False 表示在线服务场景下不打印进度条（避免刷屏）
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=settings.EMBEDDING_BATCH_SIZE,  # 默认32
                show_progress_bar=False  # 在线服务不需要显示进度条
            )
            # 【作用】将 numpy 二维数组转换成 Python 列表的列表
            # 【逻辑】每个元素是一个 1024 维的向量（List[float]），方便后续序列化
            return vectors.tolist()

        # 【作用】捕获 encode 过程中抛出的任何异常
        except Exception as e:
            # 【作用】记录错误日志
            logger.error(f"批量编码失败: {e}")
            # 【作用】让调用方知道批量编码出了问题
            raise RuntimeError(f"批量编码失败: {e}")

# 全局服务实例（单例模式）
# 【作用】定义一个模块级别的全局变量，初始值为 None
# 【逻辑】这个变量用来持有 EmbeddingService 的唯一实例，确保模型只加载一次
#        因为 BGE-M3 模型文件很大（几百 MB），重复加载会浪费内存和时间
_embedding_service = None

# 【作用】对外提供的工厂函数，获取 EmbeddingService 的单例对象
# 【返回值】EmbeddingService —— 全局唯一的 EmbeddingService 实例
def get_embedding_service() -> EmbeddingService:
    """获取Embedding服务单例（BGE-M3模型只加载一次）。"""
    # 【作用】声明 _embedding_service 是全局变量，不是局部变量
    # 【逻辑】如果没有这行声明，函数内部给 _embedding_service 赋值会创建一个新的局部变量
    #        加了 global，修改的就是模块顶层的那个 _embedding_service
    global _embedding_service
    # 【作用】判断单例是否已经创建过了
    # 【逻辑】如果为 None，说明是第一次调用，需要创建实例并赋值
    #        如果不为 None，说明之前已经创建过了，直接复用
    if _embedding_service is None:
        # 【作用】创建 EmbeddingService 实例（此处会触发模型的加载）
        # 【逻辑】构造函数 __init__ 会调用 _load_model()，这可能是耗时操作
        _embedding_service = EmbeddingService()
    # 【作用】返回全局唯一的 EmbeddingService 实例
    # 【逻辑】无论第几次调用，返回的都是同一个对象
    return _embedding_service
