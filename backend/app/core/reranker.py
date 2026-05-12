# -*- coding: utf-8 -*-
"""
重排序模块（Reranker）
功能：使用Cross-Encoder对检索结果进行精排

模块职责：
1. CrossEncoderReranker - Cross-Encoder重排序器
2. RerankerService - 重排序服务封装

为什么需要重排序？
1. 两阶段检索：先用快速的向量检索召回Top-K，再用精排选出Top-N
2. 精度提升：Cross-Encoder考虑Query和Doc的交互关系，精度更高
3. 语义理解：能够捕捉更深层的语义相似性
"""

# 导入NumPy
import numpy as np
# 导入类型注解
from typing import List, Tuple, Dict
# 导入数据类
from dataclasses import dataclass


@dataclass
class RerankResult:
    """
    重排序结果数据类
    
    存储文档重排序后的信息
    """
    document: Dict           # 原始文档
    relevance_score: float   # 相关性得分
    original_rank: int       # 原始排名
    new_rank: int          # 重排后的排名


class CrossEncoderReranker:
    """
    Cross-Encoder重排序器
    
    工作原理：
    - Bi-Encoder（双塔模型）：分别编码Query和Doc，计算余弦相似度
      优点：速度快，可预计算Doc向量
      缺点：不考虑Query-Doc交互
      
    - Cross-Encoder（交叉编码器）：将Query和Doc拼接后一起编码
      优点：考虑Query-Doc交互，精度高
      缺点：速度慢，无法预计算
    
    使用场景：
    1. 先用Bi-Encoder（向量检索）快速召回Top-K（如100个）
    2. 再用Cross-Encoder精排，选出最终的Top-N（如5个）
    
    模型选择：
    - bge-reranker-base: BGE系列重排序模型，效果好
    - msmarco-roberta-base: MS MARCO比赛冠军模型
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称或路径
        """
        self.model_name = model_name
        self.model = None      # 模型实例
        self.tokenizer = None  # 分词器
        self.device = None     # 计算设备
        self._load_model()
        
    def _load_model(self):
        """
        加载重排序模型

        使用Hugging Face Transformers加载模型和分词器，
        并自动选择GPU或CPU。
        支持 HuggingFace 和 Modelscope 两种缓存路径。
        """
        try:
            # 导入Transformer库
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            import os

            # 自动检测本地缓存路径
            hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-base")
            ms_cache = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-reranker-base")

            # 确定本地模型路径：优先 HuggingFace，其次 Modelscope
            local_path = None
            if os.path.exists(hf_cache):
                local_path = hf_cache
            elif os.path.exists(ms_cache):
                local_path = ms_cache

            if local_path is None:
                raise FileNotFoundError(
                    f"本地未找到模型缓存，请先下载: {self.model_name}"
                )

            # 使用本地路径加载，禁止联网（local_files_only=True）
            # 原理：指定精确的本地路径，from_pretrained 会直接读取该目录下
            # 的 config.json / model.safetensors / pytorch_model.bin 等文件
            print(f"从本地加载重排序模型: {local_path}")

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                local_files_only=True,
            )
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                local_path,
                local_files_only=True,
            )
            self.model.eval()  # 评估模式
            
            # 自动选择设备：优先使用GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
        except Exception as e:
            # 模型加载失败时打印警告
            print(f"Warning: Failed to load Cross-Encoder model: {e}")
            print("Falling back to simple reranking...")
            self.model = None
            
    def rerank(
        self, 
        query: str, 
        documents: List[Tuple[Dict, float]], 
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 用户查询
            documents: 候选文档列表 [(doc, original_score), ...]
            top_k: 返回前k个结果
            
        Returns:
            重排序后的结果列表
        """
        # 空文档检查
        if not documents:
            return []
            
        # 模型加载失败时使用简单重排序
        if self.model is None:
            return self._simple_rerank(query, documents, top_k)
            
        # 使用Cross-Encoder重排序
        return self._cross_encoder_rerank(query, documents, top_k)
        
    def _cross_encoder_rerank(
        self, 
        query: str, 
        documents: List[Tuple[Dict, float]], 
        top_k: int
    ) -> List[RerankResult]:
        """
        使用Cross-Encoder进行重排序
        
        流程：
        1. 准备Query-Doc对
        2. 批量编码
        3. 预测相关性分数
        4. 排序并返回结果
        """
        import torch
        
        # 准备输入对：Query-Doc
        pairs = []
        for doc, _ in documents:
            # 获取文档文本，截断到512字符（内存限制，防止单条文本过长导致GPU OOM）
            text = doc.get("content", "")[:512]
            pairs.append([query, text])
        # 批量编码
        with torch.no_grad():
            # 使用分词器对query-doc对进行批量化编码
            inputs = self.tokenizer(
                pairs,
                padding=True,          # 填充到batch内最长序列长度，使所有输入shape一致，满足GPU并行计算要求
                truncation=True,      # 超过max_length的序列自动截断尾部token，避免超出模型输入长度限制
                return_tensors="pt",   # 返回PyTorch张量格式（"pt"），可直接送入模型前向传播
                max_length=512         # 模型支持的最大序列长度（与BERT系列一致），超长文本会被截断
            )
            # 将输入移到计算设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型预测
            outputs = self.model(**inputs)
            # 获取logits（二分类输出），展平为一维向量
            scores = outputs.logits.view(-1,).float()
            # 通过sigmoid将logits映射到(0,1)区间，作为相关性概率分数
            scores = torch.sigmoid(scores).cpu().numpy()
            
        # 构建结果列表，将模型预测的sigmoid分数写入RerankResult
        results = []
        for idx, ((doc, original_score), relevance_score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                document=doc,
                relevance_score=float(relevance_score),
                original_rank=idx,      # 保留原始检索排名（升序索引）
                new_rank=0              # 占位，稍后按重排分数排序后更新
            ))
            
        # 按relevance_score降序排序：分数越高相关性越强，排在最前面
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 遍历排序后的结果，重新赋予排名（0-based）
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank
            
        return results[:top_k]
        
    def _simple_rerank(
        self, 
        query: str, 
        documents: List[Tuple[Dict, float]], 
        top_k: int
    ) -> List[RerankResult]:
        """
        简单重排序（基于关键词匹配）
        
        当Cross-Encoder模型不可用时使用，作为降级方案
        
        方法：
        1. 对Query和Doc进行分词
        2. 计算Jaccard相似度
        3. 结合原始向量检索得分
        """
        import jieba
        
        # 分词
        query_tokens = set(jieba.cut(query))
        
        results = []
        for idx, (doc, original_score) in enumerate(documents):
            # 获取文档文本
            text = doc.get("content", "")
            doc_tokens = set(jieba.cut(text))
            
            # 计算Jaccard相似度衡量query与doc的词级别重合程度
            # Jaccard = |A∩B| / |A∪B|，即交集大小除以并集大小，值域[0,1]
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            jaccard = intersection / union if union > 0 else 0
            
            # 综合得分：向量检索得分占70%（语义匹配），Jaccard相似度占30%（关键词匹配）
            # 两者加权融合后既保留语义信息，又强化了关键词命中信号的贡献
            combined_score = original_score * 0.7 + jaccard * 0.3
            
            results.append(RerankResult(
                document=doc,
                relevance_score=combined_score,
                original_rank=idx,
                new_rank=0
            ))
            
        # 排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 更新排名
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank
            
        return results[:top_k]


class RerankerService:
    """
    重排序服务
    
    对检索服务的结果进行精排，提供简洁的API接口
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序服务
        
        Args:
            model_name: 重排序模型名称
        """
        self.reranker = CrossEncoderReranker(model_name)
        
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        对文档列表进行重排序
        
        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回前k个
            
        Returns:
            重排序后的文档列表
        """
        # 转换为元组格式（doc, score）
        doc_tuples = [(doc, 0.0) for doc in documents]
        
        # 重排序
        results = self.reranker.rerank(query, doc_tuples, top_k)
        
        # 提取文档并添加分数
        reranked_docs = []
        for result in results:
            # 复制文档避免修改原数据
            doc = result.document.copy()
            # 添加重排序相关信息
            doc["rerank_score"] = result.relevance_score
            doc["original_rank"] = result.original_rank
            reranked_docs.append(doc)
            
        return reranked_docs
        
    def rerank_with_scores(
        self,
        query: str,
        documents_with_scores: List[Tuple[Dict, float]],
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        对带原始分数的文档进行重排序
        
        Args:
            query: 查询字符串
            documents_with_scores: [(doc, original_score), ...]
            top_k: 返回前k个
            
        Returns:
            [(doc, rerank_score), ...]
        """
        # 调用重排序器，返回按重排得分降序排列的RerankResult列表（已截取top_k）
        results = self.reranker.rerank(query, documents_with_scores, top_k)
        
        # 转换为(文档, 新得分)元组列表：舍弃original_rank/new_rank等内部字段，
        # 只保留调用方需要的文档对象和重排序模型产出的相关性分数
        return [(r.document, r.relevance_score) for r in results]


# 全局服务实例（单例）
_reranker_service = None

def get_reranker_service() -> RerankerService:
    """
    获取重排序服务实例（单例）
    
    Returns:
        RerankerService实例
    """
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
