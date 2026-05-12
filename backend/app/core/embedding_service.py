"""
backend/app/core/embedding_service.py — Embedding服务（单一职责）
=================================================================
<<<<<<< HEAD
职责：将文本转换为向量

功能：
  1. 加载BGE-M3模型
  2. 文本编码为1024维向量
  3. 批量编码支持

依赖：
  - sentence-transformers
  - backend.app.config.settings

被依赖：
  - backend.app.core.retrieval_service

异常：
  - 模型加载失败 → RuntimeError
  - 编码失败 → RuntimeError
=======
作用：将文本转换为向量（Embedding），是RAG系统中"理解语义"的核心
原理：
  1. 加载BGE-M3模型（多语言Embedding模型，输出1024维向量）
  2. 把"公司营收是多少"这样的文本转成一串浮点数
  3. 语义相近的文本，向量距离近；语义无关的，距离远
  4. normalize_embeddings=True：输出归一化向量，后续用内积(IP)=余弦相似度
>>>>>>> 6d37b33 (提交信息)
"""

import logging
from typing import List

<<<<<<< HEAD
from sentence_transformers import SentenceTransformer
=======
from sentence_transformers import SentenceTransformer  # 加载和使用Embedding模型
>>>>>>> 6d37b33 (提交信息)

from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
<<<<<<< HEAD
    Embedding服务
    
    封装BGE-M3模型，提供文本转向量功能
    
    使用示例：
        service = EmbeddingService()
        vector = service.encode("公司的营业收入是多少？")
    """
    
    def __init__(self):
        """
        初始化Embedding服务
        
        自动加载BGE-M3模型
        """
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        加载BGE-M3模型
        
        步骤：
          1. 检查模型路径
          2. 加载模型（自动选择CPU/GPU）
          3. 验证输出维度
=======
    作用：Embedding服务，封装BGE-M3模型，提供文本转向量功能
    使用示例：
        service = EmbeddingService()
        vector = service.encode("公司的营业收入是多少？")
        # vector = [0.012, -0.034, 0.567, ..., 0.891]  # 1024个浮点数
    """

    def __init__(self):
        """
        作用：初始化Embedding服务
        原理：自动加载BGE-M3模型，模型文件在config.EMBEDDING_MODEL_PATH
              模型加载需要几秒到几十秒（模型文件约2GB）
        """
        self.model = None  # 先占位，_load_model()里真正加载
        self._load_model()

    def _load_model(self):
        """
        作用：加载BGE-M3模型
        原理：
          1. SentenceTransformer会自动检测CPU/GPU，有GPU就用GPU，没有就用CPU
          2. 加载后用一个测试文本验证输出维度是不是1024
        逻辑：
          1. 从config读模型路径
          2. SentenceTransformer加载模型
          3. 用"测试"两个字跑一次编码，验证输出维度
          4. 维度不对抛警告（但程序继续，不影响使用）
>>>>>>> 6d37b33 (提交信息)
        """
        try:
            model_path = settings.EMBEDDING_MODEL_PATH
            logger.info(f"加载Embedding模型: {model_path}")
<<<<<<< HEAD
            
            # 加载模型
            self.model = SentenceTransformer(model_path)
            
            # 验证维度
=======

            # 加载模型
            # 原理：SentenceTransformer会自动从本地路径加载
            # 明确指定 device='cuda' 使用 GPU（如果有），加速编码
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(model_path, device=device)
            logger.info(f"Embedding模型加载完成，使用设备: {device}")

            # 验证维度
            # 作用：确保模型输出维度（1024）和config里配置的一致
            # 如果维度不一致，Milvus检索时会报错（要求向量维度精确匹配）
>>>>>>> 6d37b33 (提交信息)
            test_vector = self.model.encode("测试")
            if len(test_vector) != settings.EMBEDDING_DIM:
                logger.warning(
                    f"模型输出维度不匹配: "
                    f"期望{settings.EMBEDDING_DIM}, 实际{len(test_vector)}"
                )
<<<<<<< HEAD
            
            logger.info("Embedding模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"Embedding模型加载失败: {e}")
    
    def encode(self, text: str) -> List[float]:
        """
        将文本编码为向量
        
        参数：
            text: 输入文本
        
        返回：
            1024维浮点数向量
        
=======

            logger.info("Embedding模型加载完成")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"Embedding模型加载失败: {e}")

    def encode(self, text: str) -> List[float]:
        """
        作用：将一段文本编码为1024维向量
        原理：
          1. BGE-M3模型将文本分词 → Transformer编码 → 池化 → 1024维向量
          2. normalize_embeddings=True：向量模长归一化为1
          3. 归一化后，两个向量的内积 = 它们的余弦相似度
        参数：
            text: 输入文本，如"公司的营业收入是多少？"
        返回：
            1024个浮点数组成的列表，如 [0.012, -0.034, 0.567, ..., 0.891]
>>>>>>> 6d37b33 (提交信息)
        异常：
            RuntimeError: 编码失败
        """
        try:
<<<<<<< HEAD
            vector = self.model.encode(
                text,
                normalize_embeddings=True  # 归一化
            )
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise RuntimeError(f"文本编码失败: {e}")
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文本
        
        参数：
            texts: 文本列表
        
        返回：
            向量列表
        """
        try:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False
            )
            return vectors.tolist()
            
=======
            # encode() 返回numpy数组
            # normalize_embeddings=True → 向量归一化，内积=余弦相似度
            vector = self.model.encode(
                text,
                normalize_embeddings=True
            )
            # tolist() 把numpy数组转成Python列表，方便存JSON
            return vector.tolist()

        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise RuntimeError(f"文本编码失败: {e}")

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        作用：批量编码多段文本（比逐条encode快很多，因为可以利用GPU并行计算）
        原理：把多个文本一起送入模型，GPU可以同时处理多个，提高吞吐量
        参数：
            texts: 文本列表，如 ["营收是多少", "净利润是多少", "风险有哪些"]
        返回：
            向量列表，每个元素是1024维的浮点数列表
        """
        try:
            # batch_size参数控制一次处理多少条
            # 太小利用不了GPU并行，太大可能显存溢出
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=settings.EMBEDDING_BATCH_SIZE,  # 默认32
                show_progress_bar=False  # 在线服务不需要显示进度条
            )
            return vectors.tolist()

>>>>>>> 6d37b33 (提交信息)
        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            raise RuntimeError(f"批量编码失败: {e}")


# 全局服务实例（单例模式）
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
<<<<<<< HEAD
    获取Embedding服务单例
    
    返回：
        EmbeddingService实例
    """
    global _embedding_service #声明_embedding_service是全局变量，不声明就是局部变量
=======
    作用：获取Embedding服务单例
    原理：
      1. 用 global _embedding_service 声明要使用模块级全局变量
      2. 如果_embedding_service是None，说明第一次调用，创建实例
      3. 如果已经有实例了，直接返回
      4. 这样BGE-M3模型只会加载一次，避免重复加载（模型2GB，加载很慢）
    注意：
      这个实现不是线程安全的，如果多个请求同时进来可能重复创建。
      但实际项目中EmbeddingService在启动时就会初始化，所以一般不会出问题。
    返回：
        EmbeddingService实例
    """
    global _embedding_service  # 声明_embedding_service是全局变量（不声明就是局部变量）
>>>>>>> 6d37b33 (提交信息)
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
