# -*- coding: utf-8 -*-
"""
BM25关键词检索模块
功能：基于BM25算法的关键词检索，支持中文分词

模块职责：
1. BM25Retriever - BM25检索器，实现经典的BM25排序算法
2. HybridRetriever - 混合检索器，结合向量检索和BM25

BM25算法原理：
BM25（Best Matching 25）是一种经典的信息检索排序算法，
用于计算文档与查询之间的相关性得分。
核心公式：score(D,Q) = Σ IDF(qi) * (tf * (k1+1)) / (tf + k1*(1-b+b*|D|/avgdl))

优势：
- 对词频进行饱和处理，避免长文档过度优势
- 考虑文档长度归一化
- 计算效率高
"""

# 导入数学库，用于对数运算
import math
# 导入正则表达式，用于文本清理
import re
# 导入类型注解
from typing import List, Dict, Tuple
# 导入计数器，用于词频统计
from collections import Counter
# 导入jieba分词
import jieba


class BM25Retriever:
    """
    BM25检索器类
    
    实现经典的BM25排序算法，用于关键词检索。
    BM25是TF-IDF的改进版本，解决了词频饱和和文档长度偏差问题。
    
    核心参数：
    - k1: 控制词频饱和度，值越大词频影响越大（通常1.2-2.0）
    - b: 控制文档长度归一化，值越大长文档惩罚越重（通常0.75）
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            k1: BM25参数，控制词频饱和度。
                - 值越小，对高频词的惩罚越严格
                - 值越大，词频的影响力越大
                - 推荐范围：1.2-2.0
            b: BM25参数，控制文档长度归一化程度。
                - 值越大，对长文档的惩罚越严重
                - 推荐范围：0.5-0.75
        """
        self.k1 = k1          # 词频饱和参数
        self.b = b            # 长度归一化参数
        
        # 索引数据结构
        self.documents: List[Dict] = []          # 原始文档列表
        self.tokenized_docs: List[List[str]] = []  # 分词后的文档
        self.doc_freqs: Dict[str, int] = {}      # 文档频率（包含该词的文档数）
        self.doc_lengths: List[int] = []         # 每个文档的长度（词数）
        self.avgdl: float = 0.0                  # 平均文档长度
        self.N: int = 0                          # 文档总数
        
    def build_index(self, documents: List[Dict], text_field: str = "content"):
        """
        构建BM25索引
        
        构建索引是检索前的必要步骤，会：
        1. 对所有文档进行分词
        2. 统计每个词的文档频率
        3. 计算平均文档长度
        
        Args:
            documents: 文档列表，每个文档是字典类型
            text_field: 要索引的文本字段名，默认为"content"
        """
        self.documents = documents          # 保存原始文档
        self.N = len(documents)           # 文档总数
        
        # 初始化数据
        self.tokenized_docs = []          # 分词后的文档列表
        self.doc_lengths = []             # 文档长度列表
        term_doc_count: Dict[str, int] = {}  # 词频统计
        
        # 遍历每个文档
        for doc in documents:
            # 获取要索引的文本
            text = doc.get(text_field, "")
            # 对文本进行分词
            tokens = self._tokenize(text)
            # 保存分词结果
            self.tokenized_docs.append(tokens)
            # 记录文档长度
            self.doc_lengths.append(len(tokens))
            
            # 统计文档频率（去重后统计）
            unique_terms = set(tokens)
            for term in unique_terms:
                # 记录包含该词的文档数
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
                
        self.doc_freqs = term_doc_count  # 保存词频统计
        # 计算平均文档长度
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词
        
        使用jieba分词引擎对中文文本进行分词，
        并过滤停用词和短词。
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表，如 ["公司", "财务", "数据"]
        """
        # 清理文本：只保留中文、英文、数字
        # [^\u4e00-\u9fa5] 匹配所有非汉字的字符
        # re.sub将匹配到的字符替换为空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        
        # 使用jieba分词
        # jieba.cut返回一个生成器
        tokens = list(jieba.cut(text))
        
        # 停用词表：这些词太常见，没有区分度
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
        
        # 过滤：去空格 + 长度>1 + 非停用词
        tokens = [
            t.strip()                           # 去除空白
            for t in tokens                     # 遍历每个词
            if len(t.strip()) > 1              # 长度>1
            and t.strip() not in stop_words    # 非停用词
        ]
        
        return tokens
        
    def _calc_idf(self, term: str) -> float:
        """
        计算IDF（逆文档频率）
        
        IDF公式：IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        
        含义：
        - df（文档频率）：包含该词的文档数量
        - N（文档总数）：总文档数量
        
        作用：词出现在越少的文档中，区分度越高，IDF越大
        
        Args:
            term: 查询词
            
        Returns:
            IDF值
        """
        # 获取包含该词的文档数
        df = self.doc_freqs.get(term, 0)
        # 使用平滑IDF公式，防止df=0时IDF无穷大
        # log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        
    def _calc_bm25_score(self, doc_idx: int, query_terms: List[str]) -> float:
        """
        计算单个文档的BM25得分
        
        BM25公式：
        score = Σ IDF(qi) * (tf * (k1+1)) / (tf + k1*(1-b+b*|D|/avgdl))
        
        公式解释：
        - IDF(qi)：词qi的逆文档频率
        - tf：词在文档中的词频
        - |D|：文档长度
        - avgdl：平均文档长度
        - k1, b：调节参数
        
        Args:
            doc_idx: 文档索引
            query_terms: 查询词列表
            
        Returns:
            BM25得分（越高越相关）
        """
        score = 0.0                           # 初始化得分
        doc_tokens = self.tokenized_docs[doc_idx]  # 获取文档的分词结果
        doc_len = self.doc_lengths[doc_idx]         # 获取文档长度
        term_freq = Counter(doc_tokens)             # 统计词频
        
        # 遍历每个查询词
        for term in query_terms:
            # 如果词不在文档中，跳过
            if term not in term_freq:
                continue
                
            # 计算该词的IDF
            idf = self._calc_idf(term)
            # 获取词频
            tf = term_freq[term]
            
            # BM25核心公式
            # 分子：tf * (k1 + 1)
            # 分母：tf + k1 * (1 - b + b * |D| / avgdl)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
            
        return score
        
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        检索相关文档
        
        主检索接口，根据查询返回最相关的文档
        
        Args:
            query: 查询字符串
            top_k: 返回前k个最相关的文档
            
        Returns:
            [(文档, BM25得分), ...]，按得分降序排列
        """
        # 如果没有文档，返回空
        if not self.documents:
            return []
            
        # 对查询进行分词
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
            
        # 计算所有文档的BM25得分
        scores = []
        for idx in range(self.N):
            # 计算该文档的得分
            score = self._calc_bm25_score(idx, query_terms)
            # 只保留得分>0的文档
            if score > 0:
                scores.append((idx, score))
                
        # 按得分降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k个结果，格式为(文档, 得分)
        results = [(self.documents[idx], score) for idx, score in scores[:top_k]]
        
        return results
        
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[Dict, float]]]:
        """
        批量检索
        
        一次性处理多个查询，提高效率
        
        Args:
            queries: 查询列表
            top_k: 每个查询返回前k个结果
            
        Returns:
            每个查询的检索结果列表
        """
        # 对每个查询调用search方法
        return [self.search(q, top_k) for q in queries]


class HybridRetriever:
    """
    混合检索器
    
    结合向量检索（语义相似度）和BM25（关键词匹配）的优点，
    使用RRF（Reciprocal Rank Fusion）算法融合两种检索结果。
    
    为什么需要混合检索？
    - 向量检索：语义理解强，但对精确关键词不敏感
    - BM25检索：关键词匹配精确，但不理解语义
    - 混合检索：兼顾两者优势
    """
    
    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3):
        """
        初始化混合检索器
        
        Args:
            vector_weight: 向量检索的权重
            bm25_weight: BM25检索的权重
            注意：两个权重之和最好为1.0
        """
        self.vector_weight = vector_weight   # 向量检索权重（默认0.7）
        self.bm25_weight = bm25_weight       # BM25权重（默认0.3）
        self.bm25 = BM25Retriever()          # 创建BM25检索器实例
        
    def build_index(self, documents: List[Dict], text_field: str = "content"):
        """
        构建BM25索引
        
        注意：向量索引需要在Milvus等其他系统中构建
        
        Args:
            documents: 文档列表
            text_field: 文本字段名
        """
        self.bm25.build_index(documents, text_field)
        
    def search(
        self, 
        query: str, 
        vector_results: List[Tuple[Dict, float]], 
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        混合检索
        
        融合向量检索和BM25检索的结果
        
        Args:
            query: 查询字符串
            vector_results: 向量检索结果 [(doc, score), ...]
            top_k: 返回前k个结果
            
        Returns:
            融合后的结果，按综合得分降序排列
        """
        # Step 1: BM25检索
        # 检索2倍top_k数量的结果，留出融合空间
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        
        # Step 2: RRF融合
        fused_scores = self._rrf_fuse(vector_results, bm25_results)
        
        # Step 3: 排序并返回top_k
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        return fused_scores[:top_k]
        
    def _rrf_fuse(
        self, 
        vector_results: List[Tuple[Dict, float]], 
        bm25_results: List[Tuple[Dict, float]],
        k: int = 60
    ) -> List[Tuple[Dict, float]]:
        """
        RRF（倒数排名融合）算法
        
        RRF公式：score = Σ 1 / (k + rank)
        
        原理：
        - 不直接使用原始得分，而是使用排名
        - 排名越靠前，贡献的分数越高
        - k参数控制排名重要度，k越大，排名差异的影响越小
        
        优点：
        - 不需要归一化不同检索系统的得分
        - 对异常得分有鲁棒性
        - 实现简单效果好
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            k: RRF参数，默认60
            
        Returns:
            融合后的结果列表
        """
        # 用于存储融合后的得分
        # key: 文档ID, value: 文档对象
        doc_map: Dict[str, Dict] = {}
        # 用于存储融合得分
        # key: 文档ID, value: 融合得分
        doc_scores: Dict[str, float] = {}
        
        # 处理向量检索结果
        for rank, (doc, _) in enumerate(vector_results, start=1):
            # 生成文档唯一ID
            doc_id = doc.get("id", str(hash(str(doc))))
            # 保存文档对象
            doc_map[doc_id] = doc
            # 累加RRF得分：1 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.vector_weight / (k + rank)
            
        # 处理BM25检索结果
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = doc.get("id", str(hash(str(doc))))
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.bm25_weight / (k + rank)
            
        # 构建最终结果列表
        results = []
        for doc_id, fused_score in doc_scores.items():
            doc = doc_map[doc_id]
            results.append((doc, fused_score))
            
        return results


# 全局服务实例（单例模式）
_bm25_retriever = None
_hybrid_retriever = None

def get_bm25_retriever() -> BM25Retriever:
    """
    获取BM25检索器实例（单例）
    
    Returns:
        BM25Retriever实例
    """
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever

def get_hybrid_retriever() -> HybridRetriever:
    """
    获取混合检索器实例（单例）
    
    Returns:
        HybridRetriever实例
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
