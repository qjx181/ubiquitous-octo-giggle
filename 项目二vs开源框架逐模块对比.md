# 项目二 vs 开源 RAG 框架 —— 逐模块深度对比

> 更新日期：2026-05-27
> 对比对象：RAGFlow、Dify、FastGPT、QAnything、LightRAG、MaxKB
> 目的：面试时清晰表达"为什么不用现成框架，要自己造轮子"

---

## 一、文档解析模块（parse_pdf_mineru.py）

### 我们的实现
- 使用 MinerU（mineru>=3.1.0）的 pipeline 后端
- 自动版面分析：多栏排版检测、正确阅读顺序
- 自动丢弃页眉/页脚/页码（page_header/page_footer/page_number类型标记）
- 扫描件自动切换 OCR（基于 opendatalab/PDF-Extract-Kit-1.0）
- 表格提取为 HTML 格式，公式提取为 LaTeX
- 输出 CONTENT_LIST_V2 结构化格式，按页分组

### RAGFlow
- 同样有深度文档理解，内置 DeepDoc 模块
- 支持 OCR + 表格识别 + 布局分析
- 但使用自研的布局检测模型，不依赖 MinerU
- **区别**：RAGFlow 是平台级方案（Web UI + 多格式支持），我们聚焦招股书单一场景，解析精度更高（MinerU 的 PDF-Extract-Kit 是目前开源 PDF 解析 SOTA）

### Dify
- 文档解析较简单：调用社区解析器（PyPDF2、unstructured 等）
- **没有版面分析**，直接提取文本流
- 对多栏排版、扫描件支持差
- **差距**：Dify 的文档解析是"能用"级别，我们是"精准"级别

### FastGPT
- 基于 unstructured 库做文档解析
- 支持基本的 PDF/Word/Markdown 解析
- **没有 OCR 能力**，纯文本提取
- **差距**：面对扫描件招股书完全无法处理

### QAnything（网易有道）
- 使用自研的文档解析管线
- 支持 PDF/Word/图片/音视频多模态
- 文档解析质量较高，有布局分析
- **区别**：QAnything 是闭源方案（开源版功能受限），我们用 MinerU 完全开源

### 面试话术
> 我的文档解析用了 MinerU 的 pipeline 后端，比 Dify/FastGPT 的简单文本提取多了版面分析、OCR 和表格识别。对比 RAGFlow 的 DeepDoc，MinerU 的 PDF-Extract-Kit 是目前开源 PDF 解析的 SOTA 方案，对招股书这种复杂排版效果更好。

---

## 二、文本分块模块（chunk_text.py）

### 我们的实现
- **按章节分组**切分：利用招股说明书的章节结构（第八节 财务会计信息、第四节 风险因素等）
- chunk 大小：800 字符，overlap 150 字符
- **句子边界对齐**：不在句子中间截断，找到最近的句号/换行符
- **表格单独处理**：表格保留完整表头，不和正文混在一起
- 自动过滤页眉关键词和页码正则

### RAGFlow
- 支持多种分块策略：通用分块、按标题分块、按段落分块
- **内置 Layout-aware Chunking**：根据版面分析结果分块
- 可配置 chunk 大小和 overlap
- **对比**：RAGFlow 的分块更通用（适配多种文档类型），我们的分块更针对招股书（按章节分组 + 表格特殊处理）

### Dify
- 提供自动分块和自定义分块两种模式
- 自动模式：按固定字符数切分，**不考虑语义边界**
- 支持设置 chunk 大小和 overlap
- **差距**：Dify 的自动分块是最基础的固定窗口切分，没有按章节/按标题的智能分块

### FastGPT
- 支持手动和自动分块
- 提供 QA 拆分模式（从文档中自动提取问答对）
- **有标题层级感知**：能识别 Markdown/HTML 标题
- **对比**：FastGPT 的 QA 拆分是亮点，但对非结构化文档（如招股书）效果一般

### QAnything
- 使用自研分块策略
- 支持跨文档检索
- 分块质量较高，但具体实现未公开

### LightRAG
- **图谱增强分块**：不只切分文本，还提取实体关系构建知识图谱
- 分块后自动进行实体提取和关系抽取
- **对比**：LightRAG 的分块后处理（知识图谱构建）是我们没有的，但对招股书这种结构化程度高的文档，按章节分组更实用

### 面试话术
> 我的分块策略针对招股说明书设计：按章节分组保持语义连贯性，句子边界对齐避免信息断裂，表格单独处理保留完整表头。Dify 的自动分块只是固定窗口切分，不考虑文档结构；RAGFlow 虽然有 Layout-aware Chunking，但它是通用方案，我的分块策略更贴合招股书的章节结构。

---

## 三、向量嵌入模块（embedding_service.py）

### 我们的实现
- 模型：BGE-M3（BAAI/bge-m3），1024 维
- 使用 sentence-transformers 加载
- **归一化输出**（normalize_embeddings=True）：确保内积=余弦相似度
- 自动检测 GPU/CPU，GPU 可用时自动使用 CUDA
- 批量编码支持，batch_size=32（适配 RTX 4060 8GB 显存）

### RAGFlow
- 支持多种 Embedding 模型：BGE、Jina、OpenAI text-embedding 等
- **内置模型切换 UI**：用户可在界面上选择不同模型
- **对比**：RAGFlow 模型选择更灵活，但我们固定用 BGE-M3（中文场景 SOTA）

### Dify
- 支持 OpenAI、Cohere、BGE 等多种 Embedding
- **通过 API 调用**：Embedding 通常走云端 API（OpenAI）
- **对比**：Dify 偏向云端 Embedding（需要 API 费用），我们用本地 BGE-M3（零成本、低延迟）

### FastGPT
- 默认使用 OpenAI text-embedding-ada-002
- 支持自定义 Embedding 模型
- **对比**：FastGPT 默认走 OpenAI API（有费用+延迟），我们本地部署零成本

### QAnything
- 使用自研 Embedding 模型（基于 BERT 架构）
- **跨模态 Embedding**：文本和图片用同一个向量空间
- **对比**：QAnything 的跨模态能力是独特优势，但我们聚焦文本场景，BGE-M3 的中文效果更好

### 面试话术
> 我用的是 BGE-M3，目前中文 Embedding 的 SOTA 模型，1024 维，本地部署零成本。Dify/FastGPT 默认走 OpenAI Embedding API，有费用和网络延迟。RAGFlow 虽然支持多种模型切换，但 BGE-M3 在中文金融文档上的效果是经过验证的。

---

## 四、向量检索 + BM25 混合检索模块（retrieval.py + bm25_retriever.py）

### 我们的实现
- **Milvus 向量检索**：使用 pymilvus MilvusClient，IP（内积）索引
- **BM25 关键词检索**：自实现 BM25 算法（基于 jieba 分词）
- **混合检索策略**：向量检索 Top-K + BM25 补充，结果合并去重
- **自动探测降级**：Milvus 不可用时自动降级为纯 BM25

### RAGFlow
- 支持 Elasticsearch + 向量检索混合
- **内置混合检索**：BM25 + 向量，可配置权重
- 支持多种向量数据库：Milvus、Elasticsearch、Qdrant
- **对比**：RAGFlow 的混合检索更成熟（ES 的 BM25 实现更完善），但我们的自实现 BM25 更轻量

### Dify
- 支持多种向量数据库：Weaviate、Qdrant、Milvus、PGVector
- **有混合检索**：向量 + 全文检索，可配置权重
- **对比**：Dify 的混合检索能力与我们类似，但我们有自动探测降级机制

### FastGPT
- 基于 PostgreSQL + pgvector
- **有混合检索**：向量 + 全文检索
- **对比**：FastGPT 用 PGVector（一体化方案），我们用 Milvus（专业向量数据库，性能更好）

### QAnything
- 使用自研的检索引擎
- 支持跨文档检索
- **对比**：QAnything 的跨文档检索是亮点，我们目前只支持单文档

### LightRAG
- **图谱增强检索**：不只向量检索，还查询知识图谱中的实体关系
- 双层检索：向量相似度 + 图谱关系推理
- **对比**：LightRAG 的图谱检索是独特优势，但对招股书这种结构化文档，传统混合检索已经够用

### 面试话术
> 我实现了向量 + BM25 混合检索，向量负责语义匹配，BM25 负责关键词精确匹配。Dify/FastGPT 也有混合检索，但我的系统有自动探测降级机制——Milvus 不可用时自动降级为纯 BM25，零配置部署。对比 LightRAG 的图谱增强检索，我的方案更实用（招股书的结构化程度高，不需要图谱推理）。

---

## 五、Reranker 重排序模块（reranker.py）

### 我们的实现
- 模型：BAAI/bge-reranker-v2-m3
- **Cross-Encoder 架构**：将 Query 和 Doc 拼接后一起编码，考虑交互关系
- 工作流：Bi-Encoder（向量检索）召回 Top-K → Cross-Encoder 精排选出 Top-N
- **自动探测降级**：Reranker 不可用时直接返回原始检索结果
- 支持 GPU 加速

### RAGFlow
- **内置 Reranker**：支持 Cohere Reranker、BGE Reranker 等
- 可在 UI 上配置 Reranker 模型
- **对比**：RAGFlow 的 Reranker 支持更多模型，但我们的 Cross-Encoder 方案是标准做法

### Dify
- **支持 Reranker**：可配置 Cohere、Jina 等 Reranker API
- 通常走云端 API
- **对比**：Dify 的 Reranker 走云端（有费用），我们用本地 BGE Reranker（零成本）

### FastGPT
- **没有内置 Reranker**
- 检索结果直接返回，不做精排
- **差距**：FastGPT 缺少精排环节，检索质量不如我们

### QAnything
- 使用自研 Reranker
- 支持多路召回 + 精排
- **对比**：QAnything 的 Reranker 方案与我们类似

### MaxKB
- 支持 Reranker，可配置模型
- **对比**：MaxKB 的 Reranker 能力与我们类似

### 面试话术
> 我用 BGE-Reranker-v2-m3 做 Cross-Encoder 精排，先用向量检索召回 Top-K，再用 Cross-Encoder 精排选出 Top-N。FastGPT 没有内置 Reranker，Dify 的 Reranker 走云端 API 有费用，我用本地部署零成本。Reranker 是提升检索质量的关键环节——Cross-Encoder 考虑 Query-Doc 交互关系，精度比单纯向量相似度高很多。

---

## 六、Query 理解模块（query_understanding.py） ⭐ 核心差异化

### 我们的实现
- **意图识别**：9 种意图类型（FINANCIAL/RISK/BUSINESS/LEGAL/MANAGEMENT/COMPARISON/GREETING/UNCLEAR/OUT_OF_SCOPE）
- **实体提取**：基于 jieba 词性标注，提取 TIME/MONEY/PERCENT/PERSON 等实体
- **问题分解**：复杂问题自动拆分为多个简单子问题
- **消歧处理**：判断是否需要向用户澄清（模糊问题返回澄清提示）
- **规则 + 词典双重方案**：jieba 分词 + 关键词匹配，不依赖 LLM

### RAGFlow
- **没有 Query 理解模块**
- 用户输入直接丢给检索，不做意图识别/实体提取
- **差距**：RAGFlow 完全跳过了 Query 理解环节

### Dify
- **没有内置 Query 理解**
- 用户可以通过 Workflow 自己搭建意图识别（调用 LLM）
- **对比**：Dify 的 Workflow 可以实现类似功能，但需要用户自己搭建，不是开箱即用

### FastGPT
- **没有 Query 理解模块**
- 检索前不做任何 Query 预处理
- **差距**：FastGPT 完全没有这个能力

### QAnything
- 有基本的 Query 改写（通过 LLM）
- **没有完整的意图识别 + 实体提取**
- **对比**：QAnything 只有 Query 改写，我们的 Query 理解更全面

### LightRAG
- 有 Query 分析（通过 LLM 提取实体和关系）
- **对比**：LightRAG 的 Query 分析侧重知识图谱查询构建，我们侧重意图识别和消歧

### 面试话术（重点）
> **这是我的系统和所有开源框架最大的区别。** RAGFlow、Dify、FastGPT 全部是"用户问什么就直接检索什么"，不做任何 Query 预处理。我的系统在检索前有完整的 Query 理解层：意图识别（判断用户问的是财务/风险/业务等）、实体提取（提取时间、金额等关键实体）、消歧处理（模糊问题主动向用户澄清）。这个设计让检索更精准——比如用户问"营收多少"，系统会识别出 FINANCIAL 意图，自动把检索范围限定在财务章节，而不是全库搜索。

---

## 七、章节过滤模块（section_filter.py）

### 我们的实现
- 基于意图的章节过滤：FINANCIAL → 财务章节，RISK → 风险章节
- 关键词→章节补充映射：处理意图识别无法覆盖的具体问题
- 将检索范围从"全库搜索"缩小到"相关章节搜索"

### RAGFlow / Dify / FastGPT / QAnything
- **都没有这个功能**
- 检索范围是整个知识库，不做章节级别的过滤
- **差距**：这是针对招股说明书场景的定制优化，通用框架不会做这种事

### 面试话术
> 这是我针对招股说明书场景的定制优化。招股说明书有严格的章节结构（财务会计信息、风险因素、业务与技术等），用户问财务问题时，答案大概率在财务章节。我的系统通过意图识别自动限定检索范围，从"全库搜索"变成"相关章节搜索"，大幅提高检索精度。通用框架（RAGFlow/Dify/FastGPT）不会做这种事，因为它们需要适配各种文档类型。

---

## 八、LLM 路由模块（llm_router.py）

### 我的实现
- **多引擎支持**：SGLang（本地高性能）+ DeepSeek（云端 API）+ Ollama（本地轻量）
- **智能路由**：问候语 → Ollama（省钱），复杂问题 → SGLang > DeepSeek > Ollama
- **故障转移**：SGLang 失败 → 自动切 DeepSeek → 自动切 Ollama
- **自动探测**：启动时自动检测各引擎可用性

### RAGFlow
- 支持多种 LLM：OpenAI、Claude、本地模型等
- **通过 UI 配置**：用户选择一个 LLM，不支持多引擎自动切换
- **没有故障转移**
- **差距**：RAGFlow 是"用户选一个 LLM"，我是"系统自动选最合适的 LLM"

### Dify
- 支持多种 LLM 提供商
- **可以在 Workflow 中配置条件路由**（不同问题走不同 LLM）
- **但需要用户手动搭建**，不是开箱即用
- **对比**：Dify 的 Workflow 理论上能实现，但我的方案是自动的

### FastGPT
- 支持多种 LLM
- **没有多引擎路由**
- 用户选择一个 LLM，不支持自动切换
- **差距**：FastGPT 没有这个能力

### 面试话术
> 我实现了 LLM 智能路由：问候语走 Ollama（本地小模型，零成本），复杂问题走 SGLang（本地高性能引擎），SGLang 挂了自动切 DeepSeek（云端 API），再挂了切 Ollama。RAGFlow/FastGPT 是用户手动选一个 LLM，没有自动路由和故障转移。Dify 的 Workflow 理论上能实现，但需要用户自己搭建，我的方案是开箱即用。

---

## 九、Pipeline 架构模块（pipeline/）

### 我们的实现
- **Protocol 接口定义**（interfaces.py）：每个阶段有明确的输入/输出类型
- **Adapter 模式**（adapters.py）：将现有模块包装为标准 Pipeline 阶段
- **Pipeline 编排器**（pipeline.py）：串联 8 个阶段，统一日志和监控
- **错误隔离**：某阶段失败可以跳过或降级，不影响整体流程
- **耗时统计**：每个阶段独立计时，方便定位瓶颈

### RAGFlow
- 使用 React + TypeScript 的前端架构
- 后端使用 Python + Flask
- **没有显式的 Pipeline 模式**，各模块通过 API 直接调用
- **对比**：我们的 Pipeline 模式更规范（接口标准化、可插拔）

### Dify
- **Workflow 是核心卖点**：可视化拖拽节点，支持条件分支、并行、循环
- 支持 DAG（有向无环图）编排
- **对比**：Dify 的 Workflow 功能更强大（可视化、DAG），但我们的 Pipeline 更轻量（纯代码、无 UI 依赖）

### FastGPT
- **Flow 模式**：类似 Dify 的 Workflow，可视化节点编排
- 支持条件分支和循环
- **对比**：FastGPT 的 Flow 可视化能力更强，但我们的 Pipeline 更适合后端服务

### 面试话术
> 我用了 Protocol + Adapter 模式构建 Pipeline：每个阶段有标准接口，模块可以独立替换，新增阶段不需要改 Pipeline 代码。Dify/FastGPT 有可视化的 Workflow/Flow，功能更强但更重；我的 Pipeline 是纯代码方案，轻量可控，适合后端服务场景。

---

## 十、评估体系模块（eval/）

### 我们的实现
- **自定义 LLM 评估脚本**（eval_v2.py）：因为 RAGAS 与 langchain 生态不兼容
- **四维评估**：Faithfulness（忠实度）、Answer Relevancy（答案相关性）、Context Precision（上下文精确度）、Context Recall（上下文召回率）
- **自动运行评估**：跑完评估自动输出分数

### RAGFlow
- **内置评估功能**：支持多种评估指标
- 可视化评估报告
- **对比**：RAGFlow 的评估更成熟（UI + 多指标），但我们有针对招股书场景的定制评估

### Dify
- **没有内置评估功能**
- 需要用户自己搭建评估流程
- **差距**：Dify 缺少评估环节

### FastGPT
- 有基本的评估功能
- 支持人工标注和自动评估
- **对比**：FastGPT 的评估能力与我们类似

### 面试话术
> 我有完整的评估体系：自定义 LLM 评估脚本，四维评估（忠实度、答案相关性、上下文精确度、上下文召回率）。RAGAS 与 langchain 生态不兼容，所以我自研了评估方案。Dify 没有内置评估功能，FastGPT 的评估能力与我们类似。

---

## 总结：核心差异化（面试必背）

| 差异化点 | 我们的系统 | RAGFlow | Dify | FastGPT |
|----------|-----------|---------|------|---------|
| **Query 理解** | ✅ 意图识别+实体提取+消歧 | ❌ 没有 | ❌ 没有 | ❌ 没有 |
| **章节过滤** | ✅ 基于意图的章节限定 | ❌ | ❌ | ❌ |
| **LLM 智能路由** | ✅ 多引擎+故障转移 | ❌ 单引擎 | ⚠️ 需手动搭建 | ❌ |
| **自动探测降级** | ✅ 4 个组件自动探测 | ❌ | ❌ | ❌ |
| **Pipeline 架构** | ✅ Protocol+Adapter | ❌ 无显式Pipeline | ✅ Workflow（更强） | ✅ Flow |
| **评估体系** | ✅ 四维自定义评估 | ✅ 内置评估 | ❌ 没有 | ⚠️ 基本评估 |

### 一句话总结
> 我的系统不比平台广度（UI/权限/多模态），比核心 RAG 链路深度。Query 理解 → 章节过滤 → 混合检索 → Reranker → LLM 路由，每一步都比开源框架更精准。特别是 Query 理理解层——RAGFlow/Dify/FastGPT 全部是"问题直接丢给检索"，我的系统在检索前有意图识别、实体提取、消歧处理，这是最大的差异化。
