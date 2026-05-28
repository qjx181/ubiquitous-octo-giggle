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

# 【作用】导入数学库，用于计算对数（IDF公式中用到了log运算）
# 【原理】math.log()用于计算自然对数，BM25的IDF公式中包含对数运算
import math
# 【作用】导入正则表达式模块，用于清理文本（去除非汉字/英文/数字的字符）
# 【原理】re.sub()用正则替换掉所有干扰符号，让jieba分词更准确
import re
# 【作用】导入类型注解，让函数签名更清晰
# 【原理】Tuple[Dict, float] 表示函数返回（文档字典，得分）的元组
from typing import List, Dict, Tuple
# 【作用】从collections导入Counter类，用于统计词频
# 【原理】Counter是Python内置的高效计数器，接收列表返回{元素:出现次数}字典
from collections import Counter
# 【作用】导入jieba中文分词库
# 【原理】jieba是Python最流行的中文分词库，支持精确模式和搜索引擎模式
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
        作用：初始化BM25检索器

        Args:
            k1: BM25参数，控制词频饱和度。
                - 值越小，对高频词的惩罚越严格
                - 值越大，词频的影响力越大
                - 推荐范围：1.2-2.0
            b: BM25参数，控制文档长度归一化程度。
                - 值越大，对长文档的惩罚越严重
                - 推荐范围：0.5-0.75
        """
        # 【作用】保存词频饱和参数k1
        # 【原理】k1控制词频的"饱和曲线"：词频到达一定程度后，再增加对得分的贡献越来越小
        self.k1 = k1
        # 【作用】保存长度归一化参数b
        # 【原理】b调节文档长度对得分的影响：b=0不惩罚长文档，b=1严格按比例惩罚
        self.b = b

        # 【作用】创建空的数据结构，等待build_index()填充
        # 【逻辑】所有数据结构初始化空，索引构建完成后才会填入真实数据

        # 【作用】原始文档列表，保存完整的文档字典
        # 【格式】[{"id": 1, "content": "文本内容", ...}, ...]
        self.documents: List[Dict] = []
        # 【作用】分词后的文档列表，每个文档是一个词列表
        # 【格式】[["公司", "财务", "报告"], ["营收", "增长"], ...]
        self.tokenized_docs: List[List[str]] = []
        # 【作用】文档频率字典：记录每个词出现在多少篇文档中
        # 【格式】{"公司": 10, "财务": 5, ...} 表示"公司"这个词在10篇文档中出现过
        self.doc_freqs: Dict[str, int] = {}
        # 【作用】每篇文档的长度（词数）
        # 【格式】[12, 8, 15, ...] 第0篇12个词，第1篇8个词
        self.doc_lengths: List[int] = []
        # 【作用】所有文档的平均长度（词数）
        # 【原理】用于BM25公式中的长度归一化
        self.avgdl: float = 0.0
        # 【作用】文档总数
        self.N: int = 0

    def build_index(self, documents: List[Dict], text_field: str = "content"):
        """
        作用：构建BM25索引

        构建索引是检索前的必要步骤，会：
        1. 对所有文档进行分词
        2. 统计每个词的文档频率
        3. 计算平均文档长度

        Args:
            documents: 文档列表，每个文档是字典类型
            text_field: 要索引的文本字段名，默认为"content"
        """
        # 【作用】保存原始文档列表，后续search()返回结果时需要用到
        self.documents = documents
        # 【作用】记录文档总数
        # 【原理】N用于IDF公式中的分母计算
        self.N = len(documents)

        # 【作用】初始化数据存储
        # 【逻辑】每次重建索引时清空之前的数据
        self.tokenized_docs = []
        self.doc_lengths = []
        # 【作用】临时词频统计字典
        # 【原理】key=词，value=包含该词的文档数（一篇文档中多次出现只算1次）
        term_doc_count: Dict[str, int] = {}

        # 【作用】遍历所有文档，逐篇进行分词和统计
        for doc in documents:
            # 【作用】从文档字典中提取要索引的文本字段
            text = doc.get(text_field, "")
            # 【作用】对文本进行中文分词
            # 【原理】jieba.cut返回生成器，list(...)转为列表
            tokens = self._tokenize(text)
            # 【作用】保存分词结果
            self.tokenized_docs.append(tokens)
            # 【作用】记录该文档的长度（词数）
            self.doc_lengths.append(len(tokens))

            # 【作用】统计文档频率：该文档包含哪些不重复的词
            # 【原理】set去重：同一个词在一篇文档中出现多次，只算1次
            unique_terms = set(tokens)
            for term in unique_terms:
                # 【逻辑】累计每个词出现在多少篇不同的文档中
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        # 【作用】保存文档频率统计结果
        self.doc_freqs = term_doc_count
        # 【作用】计算所有文档的平均长度
        # 【逻辑】总词数 / 文档数，避免除以0（空文档集时avgdl=0）
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

    def _tokenize(self, text: str) -> List[str]:
        """
        作用：中文分词

        使用jieba分词引擎对中文文本进行分词，
        并过滤停用词和短词。

        Args:
            text: 输入文本

        Returns:
            分词结果列表，如 ["公司", "财务", "数据"]
        """
        # 【作用】清理文本：只保留中文、英文、数字
        # 【原理】[^\u4e00-\u9fa5a-zA-Z0-9] 匹配所有非中文/英文/数字的字符
        # re.sub将匹配到的字符替换为空格，防止标点符号干扰分词
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)

        # 【作用】使用jieba精确模式分词
        # 【原理】jieba.cut(text)用精确模式分词，尽量不做合并
        tokens = list(jieba.cut(text))

        # 【作用】定义停用词表
        # 【原理】这些词太常见（的、了、在、是...），在所有文档中都出现，没有区分度
        # 保留它们会降低BM25的准确度
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        }

        # 【作用】分词过滤
        # 【逻辑】同时做3件事：
        #   1. 去除空白字符（strip）
        #   2. 只保留长度>1的词（单字词如"的""了"没有意义）
        #   3. 过滤停用词
        tokens = [
            t.strip()
            for t in tokens
            if len(t.strip()) > 1
            and t.strip() not in stop_words
        ]

        return tokens

    def _calc_idf(self, term: str) -> float:
        """
        作用：计算IDF（逆文档频率）

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
        # 【作用】获取包含该词的文档数
        # 【逻辑】如果词不在词频字典中（所有文档都没出现），df=0
        df = self.doc_freqs.get(term, 0)
        # 【作用】使用平滑IDF公式计算
        # 【原理】平滑公式：
        #   - 防止df=0时出现log(0)导致IDF无穷大
        #   - 加0.5是平滑项，确保即使df=0也能得到合理的IDF值
        #   - 最后+1保证IDF始终为正数
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _calc_bm25_score(self, doc_idx: int, query_terms: List[str]) -> float:
        """
        作用：计算单个文档的BM25得分

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
        # 【作用】初始化得分为0
        score = 0.0
        # 【作用】获取该文档的分词结果
        doc_tokens = self.tokenized_docs[doc_idx]
        # 【作用】获取该文档的长度（词数）
        doc_len = self.doc_lengths[doc_idx]
        # 【作用】用Counter统计文档中每个词的出现次数
        # 【原理】Counter返回{词: 出现次数}字典，用于获取词频tf
        term_freq = Counter(doc_tokens)

        # 【作用】遍历查询中的每个词，累加BM25得分
        for term in query_terms:
            # 【作用】如果该词不在文档中，跳过（得分为0）
            if term not in term_freq:
                continue

            # 【作用】计算该词的IDF（逆文档频率）
            # 【原理】IDF越大说明词越稀有、越有区分度
            idf = self._calc_idf(term)
            # 【作用】获取该词在文档中的词频（出现次数）
            tf = term_freq[term]

            # 【作用】BM25核心公式的"词频饱和"部分
            # 【原理】
            #   分子：tf * (k1 + 1) → tf越大分子越大，但增长会变慢
            #   分母：tf + k1 * (1 - b + b * |D| / avgdl)
            #     - 分母中的k1*(1-b+b*|D|/avgdl) 是长度归一化项
            #     - 文档越长，分母越大，得分惩罚越重
            #     - b控制惩罚力度
            # 【效果】这就是BM25比TF-IDF好的核心原因：
            #   - TF-IDF中tf是线性的，高频词会过度主导
            #   - BM25中tf是饱和的，高频词的影响有上限
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            # 【作用】累加该词的BM25贡献
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        作用：检索相关文档

        主检索接口，根据查询返回最相关的文档

        Args:
            query: 查询字符串
            top_k: 返回前k个最相关的文档

        Returns:
            [(文档, BM25得分), ...]，按得分降序排列
        """
        # 【作用】如果没有文档，直接返回空列表
        if not self.documents:
            return []

        # 【作用】对查询进行分词（和文档同样的分词方式）
        query_terms = self._tokenize(query)
        # 【作用】如果查询分词后没有有效词（全是停用词），返回空
        if not query_terms:
            return []

        # 【作用】计算所有文档的BM25得分
        # 【逻辑】遍历每篇文档，计算得分，只保留得分>0的结果
        scores = []
        for idx in range(self.N):
            score = self._calc_bm25_score(idx, query_terms)
            if score > 0:
                scores.append((idx, score))

        # 【作用】按BM25得分从高到低排序
        # 【原理】sort的reverse=True表示降序排列（高分在前）
        scores.sort(key=lambda x: x[1], reverse=True)

        # 【作用】截取前top_k个结果，返回(文档, 得分)格式
        # 【逻辑】根据文档索引从self.documents中取出原始文档
        results = [(self.documents[idx], score) for idx, score in scores[:top_k]]

        return results

    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[Dict, float]]]:
        """
        作用：批量检索

        一次性处理多个查询，提高效率

        Args:
            queries: 查询列表
            top_k: 每个查询返回前k个结果

        Returns:
            每个查询的检索结果列表
        """
        # 【作用】对每个查询调用search方法，返回所有结果组成的列表
        return [self.search(q, top_k) for q in queries]


class HybridRetriever:
    """
    作用：混合检索器
    原理：结合向量检索（语义相似度）和BM25（关键词匹配）的优点，
          使用RRF（Reciprocal Rank Fusion）算法融合两种检索结果。
    为什么需要混合检索？
    - 向量检索：语义理解强，但对精确关键词不敏感
    - BM25检索：关键词匹配精确，但不理解语义
    - 混合检索：兼顾两者优势

    融合方式：RRF（倒数排名融合），不是简单的分数加权。
    原因：向量检索的分数（余弦相似度0~1）和BM25的分数（理论无上限）
          不在一个量级，直接加权没有意义。RRF只看排名，不看分数绝对值。

    参数：vector_weight=0.7, bm25_weight=0.3, RRF的k=60

    边界情况：
      - BM25检索结果为空（文档太短或全是停用词）：只保留向量检索结果
      - 向量检索结果为空：直接返回BM25结果
      - 两者都为空：返回空列表
      - 文档hash碰撞（doc_id用str(hash(str(doc)))）：概率极低，可忽略

    面试官可能问：
      Q: 为什么用RRF融合而不是分数加权？
      A: BM25得分和向量余弦相似度不在一个量级。BM25得分可以到几十，
         余弦相似度在0~1之间。直接加权（0.3*bm25 + 0.7*vector）时，
         BM25的得分会压过向量得分，相当于纯BM25。
         RRF用排名代替分数：1/(k+rank)把排名统一到(0, 1/k]区间，
         解决了分数尺度不一致的问题。

      Q: RRF的k=60怎么确定的？
      A: k控制排名差异的影响程度。k越小，排名靠前的文档获得越高的
         权重增益；k越大，各排名之间的差异越小。
         k=60是业界常用值。如果希望top-1的优势更明显，可以降低k；
         如果希望更多文档有机会进入top-k，可以提高k。

      Q: BM25权重0.3/向量0.7基于什么？
      A: 大多数场景下语义匹配更重要（招股说明书中的术语多变），
         所以向量权重大。BM25作为补充，权重低防止它"带偏"结果。
         最优值应该通过消融实验确定。

      Q: BM25在RAG中还有什么应用场景？
      A: (1) 精确ID匹配 - "1-1-1"这类代码向量检索容易混淆
         (2) 罕见专业术语 - 如"加权平均净资产收益率"
         (3) 否定表达 - "不包含在合并报表范围内"
         (4) 短查询精确匹配 - 如查"第15页"
    """

    # 【作用】定义向量检索和BM25的融合权重
    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3):
        """
        作用：初始化混合检索器

        Args:
            vector_weight: 向量检索的权重（默认0.7，偏向语义匹配）
            bm25_weight: BM25检索的权重（默认0.3，辅助关键词匹配）
        """
        # 【作用】保存向量检索的融合权重
        self.vector_weight = vector_weight
        # 【作用】保存BM25检索的融合权重
        self.bm25_weight = bm25_weight
        # 【作用】创建一个BM25检索器实例，用于关键词检索
        self.bm25 = BM25Retriever()

    def build_index(self, documents: List[Dict], text_field: str = "content"):
        """
        作用：构建BM25索引
        注意：向量索引需要在Milvus等其他系统中构建

        Args:
            documents: 文档列表
            text_field: 文本字段名
        """
        # 【作用】调用内部BM25Retriever的build_index方法
        self.bm25.build_index(documents, text_field)

    def search(
        self,
        query: str,
        vector_results: List[Tuple[Dict, float]],
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        作用：混合检索
        融合向量检索和BM25检索的结果

        Args:
            query: 查询字符串
            vector_results: 向量检索结果 [(doc, score), ...]
            top_k: 返回前k个结果

        Returns:
            融合后的结果，按综合得分降序排列
        """
        # 【作用】Step 1: 用BM25检索，多召回一些留给RRF融合
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # 【作用】Step 2: RRF融合（倒数排名融合）
        fused_scores = self._rrf_fuse(vector_results, bm25_results)

        # 【作用】Step 3: 按融合得分降序排列
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        return fused_scores[:top_k]

    def _rrf_fuse(
        self,
        vector_results: List[Tuple[Dict, float]],
        bm25_results: List[Tuple[Dict, float]],
        k: int = 60
    ) -> List[Tuple[Dict, float]]:
        """
        作用：RRF（倒数排名融合）算法

        RRF公式：score = Σ 1 / (k + rank)

        原理：
        - 不直接使用原始得分，而是使用排名
        - 排名越靠前，贡献的分数越高
        - k参数控制排名重要度，k越大，排名差异的影响越小

        优点：
        - 不需要归一化不同检索系统的得分
        - 对异常得分有鲁棒性
        - 实现简单效果好
        """
        # 【作用】用字典保存 {(文档ID): 文档对象}
        doc_map: Dict[str, Dict] = {}
        # 【作用】用字典保存 {(文档ID): 融合得分}
        doc_scores: Dict[str, float] = {}

        # 【作用】处理向量检索结果：按排名计算RRF得分
        for rank, (doc, _) in enumerate(vector_results, start=1):
            # 【作用】用文档id字段作为唯一标识，没有则用hash
            doc_id = doc.get("id", str(hash(str(doc))))
            doc_map[doc_id] = doc
            # 【作用】RRF得分 = vector_weight / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.vector_weight / (k + rank)

        # 【作用】处理BM25检索结果（同上）
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = doc.get("id", str(hash(str(doc))))
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.bm25_weight / (k + rank)

        # 【作用】构建最终结果列表
        results = []
        for doc_id, fused_score in doc_scores.items():
            doc = doc_map[doc_id]
            results.append((doc, fused_score))

        return results


# 【作用】全局混合检索器实例（单例模式）
_hybrid_retriever = None


def get_hybrid_retriever() -> HybridRetriever:
    """
    作用：获取混合检索器实例（单例）

    Returns:
        HybridRetriever实例
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
