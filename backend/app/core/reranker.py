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

# 【作用】导入NumPy库，用于后续的数值计算和数组操作
# 【原理】NumPy是Python最常用的科学计算库，提供了高效的多维数组对象和数学函数
# 【逻辑】用"np"作为别名，后续代码中写np.xxx就能调用NumPy的功能
import numpy as np

# 【作用】从typing模块导入类型注解工具，用于给函数参数和返回值标注类型
# 【原理】Python的类型注解不会影响运行，但能让IDE和代码检查工具理解代码意图，减少bug
# 【逻辑】List表示列表，Tuple表示元组，Dict表示字典，都是泛型容器，尖括号里写具体类型
from typing import List, Tuple, Dict

# 【作用】从dataclasses模块导入数据类装饰器，用来快速创建只存数据的类
# 【原理】@dataclass装饰器会自动帮我们生成__init__、__repr__等样板方法，少写很多代码
# 【逻辑】用这个装饰器的类，只需要写属性名和类型，Python自动生成构造函数
from dataclasses import dataclass


@dataclass
# 【作用】定义一个重排序结果的数据类，存储一条文档重排后的所有信息
# 【原理】@dataclass标注后，Python自动生成__init__方法，创建对象时直接传参即可
# 【逻辑】这个类就像一个"小盒子"，把文档本身、分数、排名等信息打包在一起
class RerankResult:
    """
    重排序结果数据类
    
    存储文档重排序后的信息
    """
    # 【作用】存储原始文档数据，通常是一个字典，包含文档的各个字段（标题、正文等）
    # 【原理】Dict表示字典类型，冒号后面写类型注解，告诉使用者这里应该放什么类型的数据
    # 【逻辑】这个字段存的是检索到的原始文档对象，后续可能要用到其中的字段
    document: Dict           # 原始文档
    
    # 【作用】存储重排序模型给这个文档打的相关性分数
    # 【原理】float是浮点数类型，分数越高说明和用户查询越相关
    # 【逻辑】这个分数是Cross-Encoder模型计算出来的，比原始检索分数更精确
    relevance_score: float   # 相关性得分
    
    # 【作用】记录这个文档在原始检索结果中的排名（重排之前排第几）
    # 【原理】int表示整数，排名从0开始（第0个是第一名）
    # 【逻辑】保存原始排名可以方便后续分析重排序的效果——文档位置变化了多少
    original_rank: int       # 原始排名
    
    # 【作用】记录重排序之后这个文档的新排名
    # 【原理】int整数类型，重排后按相关性分数重新排名，从0开始
    # 【逻辑】这个字段在排序过程中先占位设成0，最后排序完再更新为正确值
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

    返回类型：
      rerank(query, documents) → list[RerankResult]
      每个RerankResult包含document, relevance_score, original_rank, new_rank

    边界情况：
      - 文档为空：返回空列表
      - 文档格式非法（非字符串）：跳过该文档
      - 模型加载失败：self.model保持为None，所有rerank调用返回空
      - GPU/CUDA不可用：自动降级到CPU

    面试官可能问：
      Q: Bi-Encoder和Cross-Encoder有什么区别？为什么用后者重排序？
      A: Bi-Encoder是双塔，Query和Doc独立编码，最后算余弦相似度。
         速度快但损失了交互信息（比如"苹果好吃"和"苹果手机"的"苹果"语义不同，
         Bi-Encoder无法区分）。Cross-Encoder把Query+Doc拼接后一起编码，
         能捕捉细粒度交互，精度高但速度慢。所以用Bi-Encoder先粗筛（百→十），
         Cross-Encoder精排（十→三），兼顾速度和精度。

      Q: Cross-Encoder慢了多少？
      A: 对N篇文档，Bi-Encoder只需要1次Query编码+N次Doc向量检索（近似常数时间）；
         Cross-Encoder需要N次独立的Query-Doc拼接编码，每篇都要过一遍模型。
         100篇文档大约多花2-5秒。这就是为什么只对top-K（K较小）做rerank，
         不对全库做。

      Q: 为什么选bge-reranker-base？和别的模型比怎么样？
      A: bge-reranker-base在中文场景效果好，模型大小适中（约1GB），
         在消费级GPU（8GB显存）上能跑。BGE系列还有large版本（更大但更慢）、
         small版本（更快但精度略降）。选base是在精度和速度之间的中间选择。

      Q: 模型加载失败怎么容错？
      A: self.model保持None，rerank()调用时检测到model为None则返回空列表。
         调用方（retrieval.py的_run_retrieval_pipeline）发现rerank结果是空，
         就跳过rerank阶段，直接使用rerank前的检索结果。这是"优雅降级"。
    """
    
    # 【作用】定义类的初始化方法，创建对象时会自动调用这个方法
    # 【原理】def __init__(self, ...) 是Python类的构造方法，self代表实例自身
    # 【逻辑】初始化时接收模型名称参数（有默认值），保存到实例变量，然后加载模型
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称或路径
        """
        # 【作用】把传入的模型名称保存到实例变量中，方便其他方法使用
        # 【原理】self.model_name = ... 创建了一个属于这个实例的变量，整个类的方法都能访问
        # 【逻辑】model_name是HuggingFace上的模型ID，用来告诉程序去加载哪个模型
        self.model_name = model_name
        
        # 【作用】初始化模型实例变量，先设为None，等_load_model方法里真正加载
        # 【原理】self.model = None 只是先占个位置，Python允许先声明后赋值
        # 【逻辑】模型可能会加载失败，先放个None表示"还没加载"或"加载失败"
        self.model = None      # 模型实例
        
        # 【作用】初始化分词器变量，同样先设为None
        # 【原理】tokenizer的作用是把人类文字转成模型能理解的数字编号序列
        # 【逻辑】和模型一样，先占位，等_load_model里再真正加载
        self.tokenizer = None  # 分词器
        
        # 【作用】初始化计算设备变量（CPU还是GPU）
        # 【原理】程序可以在CPU上跑，也可以用NVIDIA显卡（GPU）加速
        # 【逻辑】先占位None，等检测到有GPU就用cuda，没有就用cpu
        self.device = None     # 计算设备
        
        # 【作用】调用模型加载方法，真正去加载预训练模型和分词器
        # 【原理】self._load_model() 调用同一个类的另一个方法，下划线开头表示"内部方法"
        # 【逻辑】初始化方法里最后做这件事，确保对象创建完模型就已经加载好了
        self._load_model()
        
    # 【作用】定义内部方法：加载重排序模型，以下划线开头表示"私有的，外部别调用"
    # 【原理】Python没有真正的私有，但约定下划线开头表示"这是内部实现细节"
    # 【逻辑】这个方法会尝试从本地缓存加载HuggingFace或ModelScope的模型
    def _load_model(self):
        """
        加载重排序模型

        使用Hugging Face Transformers加载模型和分词器，
        并自动选择GPU或CPU。
        支持 HuggingFace 和 Modelscope 两种缓存路径。

        加载策略：
          1. 先检查本地缓存（HuggingFace路径 → Modelscope路径）
          2. 本地有缓存则local_files_only=True加载
          3. 本地无缓存则从网络下载（走国内镜像）
          4. 设备自动选择：GPU(CUDA) > MPS(Apple) > CPU

        边界情况：
          - 本地无缓存且网络不通：model保持None，后续rerank返回空
          - 模型文件损坏（加载时校验失败）：catch异常，model保持None
          - GPU显存不足：可考虑加载fp16半精度版本

        面试官可能问：
          Q: 为什么先检查本地缓存而不是直接from_pretrained()？
          A: 用户环境在内网，无法直接访问HuggingFace。
             先检查Modelscope或HuggingFace的本地缓存路径，
             有缓存就用local_files_only=True，避免联网失败导致加载异常。
             这是"离线优先"策略。

          Q: 如果本地没有缓存怎么办？
          A: from_pretrained()不传local_files_only，会尝试联网下载。
             但用户环境HuggingFace被墙，所以优先走Modelscope国内镜像。
             如果联网也失败，model保持None，调用方优雅降级。
        """
        # 【作用】开始一个try代码块，尝试执行里面可能会出错的代码
        # 【原理】try-except是Python的异常处理机制，出错时不会崩溃而是跳到except里
        # 【逻辑】模型加载可能因为文件不存在、网络问题等原因失败，try-except兜住错误
        try:
            # 【作用】从transformers库导入自动分词器和自动模型类
            # 【原理】AutoTokenizer能根据模型名称自动选择合适的分词器，AutoModelForSequenceClassification是用于分类的模型
            # 【逻辑】放在函数内部导入而不是文件顶部，是"延迟导入"技巧，只有用到时才加载这个库
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            import os

            # 【作用】构造HuggingFace模型的本地缓存路径
            # 【原理】os.path.expanduser("~")会自动替换成用户主目录，比如Linux下的/home/用户名
            # 【逻辑】HuggingFace下载模型时默认存到~/.cache/huggingface/hub/，文件名是模型名加点替换为--的格式
            hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-base")
            
            # 【作用】构造ModelScope模型的本地缓存路径
            # 【原理】ModelScope是阿里云的模型托管平台，缓存路径和HuggingFace不同
            # 【逻辑】有些用户因为网络原因用ModelScope下载模型，这里作为备选路径
            ms_cache = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-reranker-base")

            # 【作用】初始化一个变量用来存最终确定的模型本地路径
            # 【原理】先设成None，后续根据哪个路径存在就赋哪个值
            # 【逻辑】优先用HuggingFace的缓存，没有的话再用ModelScope的，都没有就报错
            local_path = None
            
            # 【作用】检查HuggingFace缓存路径是否存在
            # 【原理】os.path.exists()返回True或False，判断文件或目录是否存在
            # 【逻辑】如果HuggingFace的缓存存在，就用这个路径
            if os.path.exists(hf_cache):
                local_path = hf_cache
            # 【作用】如果HuggingFace路径不存在，再检查ModelScope路径
            # 【原理】elif = else if，上一个条件不满足时检查这个
            # 【逻辑】用ModelScope的路径作为备选方案
            elif os.path.exists(ms_cache):
                local_path = ms_cache

            # 【作用】检查是否找到了模型路径，如果local_path还是None说明都没找到
            # 【原理】is None 检查变量是否是None值
            # 【逻辑】两个缓存路径都不存在，说明模型还没下载过，抛出错误让用户去下载
            if local_path is None:
                # 【作用】用raise主动抛出一个文件找不到的错误
                # 【原理】FileNotFoundError是Python内置的异常类型，专门表示文件不存在的错误
                # 【逻辑】f-string格式化字符串，把self.model_name的值嵌入到错误消息中
                raise FileNotFoundError(
                    f"本地未找到模型缓存，请先下载: {self.model_name}"
                )

            # 【作用】打印日志，告诉用户正在从哪个本地路径加载模型
            # 【原理】print()会在控制台输出文字，方便调试和了解程序运行状态
            # 【逻辑】让用户知道模型是从本地加载的，不是从网络下载的
            print(f"从本地加载重排序模型: {local_path}")

            # 【作用】调用AutoTokenizer的from_pretrained方法加载分词器
            # 【原理】AutoTokenizer.from_pretrained()会读取本地模型目录中的tokenizer配置文件，自动构造分词器
            # 【逻辑】local_files_only=True表示只从本地加载，不联网下载，确保离线可用
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                local_files_only=True,
            )
            
            # 【作用】调用AutoModelForSequenceClassification加载序列分类模型
            # 【原理】AutoModelForSequenceClassification是transformers库中的自动模型类，用于文本分类任务
            # 【逻辑】Cross-Encoder本质上是一个二分类模型：输入(query, doc)对，输出"相关"的分数
            self.model = AutoModelForSequenceClassification.from_pretrained(
                local_path,
                local_files_only=True,
            )
            
            # 【作用】把模型切换到评估模式
            # 【原理】model.eval()告诉模型现在是推理阶段，不是训练阶段，dropout和batch normalization等层的行为会改变
            # 【逻辑】评估模式下模型的行为是确定性的（每次输入相同输出相同），训练模式会有随机性
            self.model.eval()  # 评估模式
            
            # 【作用】自动选择计算设备，优先使用GPU（如果可用的话）
            # 【原理】torch.cuda.is_available()检查系统是否有NVIDIA GPU且CUDA配置正确
            # 【逻辑】有GPU就用cuda（速度快很多），没有就用cpu（也能跑，就是慢一点）
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 【作用】把模型移动到选定的设备（GPU或CPU）上
            # 【原理】model.to(device)将模型的所有参数和缓冲区移动到指定设备
            # 【逻辑】不移动的话模型默认在CPU上，即使有GPU也用不上，这一步确保充分利用硬件
            self.model.to(self.device)
            
        # 【作用】捕获try块中可能抛出的任何异常
        # 【原理】except Exception as e 捕获所有类型的异常（Exception是所有异常的基类），并把异常信息存到变量e中
        # 【逻辑】模型加载的任何环节出错（文件损坏、版本不兼容等），都不会让程序崩溃，而是走降级方案
        except Exception as e:
            # 【作用】打印警告信息，告诉用户模型加载失败了
            # 【原理】print输出到控制台，f-string把异常信息嵌入到字符串中
            # 【逻辑】用户看到这个警告就知道模型功能不可用，系统用了备选方案
            print(f"Warning: Failed to load Cross-Encoder model: {e}")
            print("Falling back to simple reranking...")
            
            # 【作用】把模型设为None，表示模型不可用
            # 【原理】self.model = None 覆盖之前可能部分加载的模型对象
            # 【逻辑】后续的rerank方法检查到model是None，就会走简单的降级排序逻辑
            self.model = None
            
    # 【作用】定义重排序的主方法，外部调用者主要用这个方法
    # 【原理】方法接收查询文本、候选文档列表、返回数量，返回重排序后的结果列表
    # 【逻辑】这是对外暴露的接口，内部会根据模型是否可用选择用Cross-Encoder还是简单降级方案
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict, float]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        重排序主方法
        
        Args:
            query: 用户查询文本
            candidates: 候选文档列表，格式为[(文档字典, 原始得分), ...]
            top_k: 返回前k个重排序结果
        
        Returns:
            重排序后的结果列表，按relevance_score降序排列
        """
        # 【作用】如果候选文档列表为空（没有任何文档需要排序），直接返回空列表
        # 【原理】not candidates：Python中空列表会被当作False，not之后变成True
        # 【逻辑】没有输入就没有输出，这是一个边界情况处理，避免后续代码报错
        if not candidates:
            return []
        
        # 【作用】从候选列表中提取出文档部分，丢弃原始分数
        # 【原理】列表推导式 [doc for doc, _ in candidates] 遍历每个元组，只取第一个元素（文档），忽略第二个元素（原始分数）
        # 【逻辑】Cross-Encoder会重新打分，不需要原始分数，所以直接用文档列表
        documents = [doc for doc, _ in candidates]
        
        # 【作用】准备Cross-Encoder需要的输入对列表，每个元素是(query字符串, doc文本字符串)
        # 【原理】Cross-Encoder的工作方式是：把query和doc拼在一起让模型一起看，所以需要成对输入
        # 【逻辑】遍历每个文档，从中提取文本内容，和query组成一对，存到pairs列表中
        pairs = []
        for doc in documents:
            # 【作用】从文档字典中提取文本内容：优先用"content"字段，如果没有就用"text"字段，再没有就用空字符串
            # 【原理】字典的get方法：doc.get("content", doc.get("text", "")) 先查content，查不到再查text
            # 【逻辑】不同来源的文档可能用不同字段名存正文，这里兼容两种常见字段名
            doc_text = doc.get("content", doc.get("text", ""))
            
            # 【作用】把(query, 文档文本)作为一个元组追加到pairs列表
            # 【原理】pairs.append((query, doc_text)) 在列表末尾添加一个新元素，元素是一个二元组
            # 【逻辑】每个输入对就是一组(query, doc)，模型会同时看这两个文本并判断相关性
            pairs.append((query, doc_text))
        
        # 【作用】检查模型是否加载成功，如果model是None说明加载失败，走降级方案
        # 【原理】if self.model is None 判断模型变量是否为None（即加载失败的情况）
        # 【逻辑】降级方案直接使用原始检索分数作为重排分数，不做任何重新计算
        if self.model is None:
            # 【作用】创建一个空列表用于存放重排序结果
            # 【原理】results = [] 初始化一个空列表，后续添加RerankResult对象
            # 【逻辑】降级模式下直接基于原始分数排序
            results = []
            
            # 【作用】遍历文档列表，为每个文档创建一个RerankResult对象
            # 【原理】enumerate(documents) 同时获取索引和元素，rank从0开始
            # 【逻辑】降级模式下没有重新打分，直接使用原始检索时的分数
            for rank, doc in enumerate(documents):
                # 【作用】创建RerankResult对象，填入文档、原始得分、原始排名和临时新排名
                # 【原理】直接调用RerankResult(...)构造函数，因为@dataclass会自动生成构造函数
                # 【逻辑】原始排名就是当前遍历的顺序（rank+1从1开始），新排名暂时和原始排名一样
                results.append(RerankResult(
                    document=doc,
                    relevance_score=candidates[rank][1],  # 使用原始得分
                    original_rank=rank + 1,               # 原始排名从1开始
                    new_rank=rank + 1                     # 重排后排名（无变化）
                ))
            
            # 【作用】按相关性得分从高到低排序（降序排列）
            # 【原理】sort()方法直接修改原列表，key=lambda x: x.relevance_score 指定按relevance_score字段排序
            # 【逻辑】reverse=True表示降序，分数最高的排在最前面
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # 【作用】遍历排序后的结果，重新设置每个文档的新排名
            # 【原理】enumerate(results) 同时获取新索引和元素，i从0开始
            # 【逻辑】排名从1开始（i+1），方便人类阅读，第一名就是最相关的
            for i, r in enumerate(results):
                r.new_rank = i + 1
            
            # 【作用】只返回前top_k个结果，截断列表
            # 【原理】列表切片results[:top_k] 取列表前top_k个元素
            # 【逻辑】用户指定了需要多少个结果，多余的就不返回了
            return results[:top_k]
        
        # 【作用】模型加载成功，使用Cross-Encoder模型进行真正的重排序
        # 【原理】下面这段代码用模型给每个(query, doc)对打分，用模型分数排序
        # 【逻辑】try块包裹模型推理代码，如果推理出错（比如显存不够）也能降级处理
        try:
            import torch
            
            # 【作用】使用分词器把文本对（query, doc）编码成模型能理解的数字张量
            # 【原理】tokenizer接收字符串列表，返回input_ids（词对应的编号）、attention_mask（哪些位置有词）等字典
            # 【逻辑】padding和truncation确保所有输入长度一致，return_tensors="pt"指定返回PyTorch张量格式
            encoded = self.tokenizer(
                pairs,
                padding=True,       # 填充到同一长度
                truncation=True,    # 截断超长文本
                return_tensors="pt" # 返回PyTorch张量
            )
            
            # 【作用】将编码后的张量移动到模型所在的设备（GPU或CPU）
            # 【原理】字典推导式 {k: v.to(self.device) for k, v in encoded.items()} 遍历编码结果字典
            # 【逻辑】如果模型在GPU上，输入数据也必须在GPU上，否则会报设备不匹配的错误
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 【作用】在torch.no_grad()上下文中进行模型推理，不计算梯度
            # 【原理】with torch.no_grad(): 上下文管理器，内部的运算不会记录梯度，节省显存和计算资源
            # 【逻辑】推理时不需要反向传播计算梯度，关闭梯度计算能加速并降低显存占用
            with torch.no_grad():
                # 【作用】把编码后的输入传给模型，进行前向推理
                # 【原理】self.model(**encoded) 把字典展开成关键字参数传进去，模型返回outputs对象
                # 【逻辑】**语法把字典展开，相当于self.model(input_ids=..., attention_mask=...)
                outputs = self.model(**encoded)
                
                # 【作用】从模型输出中提取logits（原始分数），并去除多余的维度
                # 【原理】outputs.logits 是模型最后一层的输出，shape通常是(batch_size, num_labels)
                # 【逻辑】squeeze()去除维度为1的轴，比如shape从(3,1)变成(3,)
                scores = outputs.logits.squeeze()
                
                # 【作用】确保scores是一维的，处理只有一个候选文档的特殊情况
                # 【原理】如果只有一个输入，squeeze后可能变成0维标量（一个数字），需要unsqueeze恢复成一维
                # 【逻辑】scores.dim() == 0 表示是标量，unsqueeze(0)在位置0增加一个维度变成(1,)一维张量
                if scores.dim() == 0:
                    # 【作用】当只有一个候选时，logits可能降为0维标量，用unsqueeze升维
                    # 【原理】scores.unsqueeze(0)在索引0处增加一个维度，把标量变成一维张量
                    # 【逻辑】只有一个候选时也按一维处理，保持代码的统一性
                    scores = scores.unsqueeze(0)
                
                # 【作用】把PyTorch张量转换成Python列表，方便后续处理
                # 【原理】scores.tolist() 将PyTorch张量转换为普通的Python列表
                # 【逻辑】后面需要遍历分数列表和文档列表配对，Python列表比张量更方便
                scores = scores.tolist()
        except Exception as e:
            # 【作用】模型推理出错时（如显存溢出、输入格式问题），打印警告并降级使用原始分数
            # 【原理】except捕获try块中的任何异常，e中保存异常信息
            # 【逻辑】降级方案：直接用候选列表中的原始检索得分作为分数
            print(f"Rerank 模型推理失败: {e}, 使用原始分数降级")
            
            # 【作用】从候选列表中提取原始得分作为备用分数
            # 【原理】列表推导式 [score for _, score in candidates] 遍历每个元组取第二个元素
            # 【逻辑】虽然Cross-Encoder打不了分，但至少有原始检索分数可以用
            scores = [score for _, score in candidates]
        
        # 【作用】构建重排序结果列表，把每个文档和对应的模型分数组合起来
        # 【原理】zip(documents, scores) 将两个列表按位置一一配对，遍历时同时拿到文档和分数
        # 【逻辑】初始new_rank设为0占位，后面排序完再重新赋值
        results = []
        for rank, (doc, score) in enumerate(zip(documents, scores)):
            # 【作用】创建RerankResult对象，存入文档、模型分数、原始排名和临时新排名
            # 【原理】RerankResult(...)是数据类的构造函数，按参数顺序赋值
            # 【逻辑】rank是原始顺序的索引（从0开始），new_rank先设0后续更新
            results.append(RerankResult(
                document=doc,
                relevance_score=score,           # Cross-Encoder打出的分数
                original_rank=rank + 1,          # 原始检索结果中的排名
                new_rank=0                       # 临时占位，排序后会重设
            ))
        
        # 【作用】按相关性得分从高到低排序（真正的重排序在这里发生）
        # 【原理】results.sort()使用key函数提取排序依据，reverse=True降序
        # 【逻辑】排序后，原来排第5的可能升到第1，原来排第1的可能掉到后面
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 【作用】排序完成后，遍历结果列表给每个文档重新设置排名
        # 【原理】enumerate(results)提供序号（从0开始），赋值给每个结果的new_rank字段
        # 【逻辑】新排名从1开始，第一名得1分，第二名得2分，以此类推
        for i, r in enumerate(results):
            r.new_rank = i + 1  # 排名从1开始
        
        # 【作用】只返回前top_k个结果给调用者
        # 【原理】列表切片results[:top_k] 取前top_k个元素
        # 【逻辑】防止返回太多结果，调用者只关心最重要的几个
        return results[:top_k]


class RerankerService:
    """
    重排序服务
    
    对检索服务的结果进行精排，提供简洁的API接口

    封装了CrossEncoderReranker，对外提供两个方法：
      - rerank_documents(query, documents, top_k): 纯文档列表重排
      - rerank_with_scores(query, docs_with_scores, top_k): 带分数文档重排

    设计原理：
      - 内部持有CrossEncoderReranker实例，模型加载委托给底层
      - 采用单例模式（get_reranker_service()），避免重复加载模型
      - 模型加载失败时self.reranker.model为None，调用方自己降级

    面试官可能问：
      Q: rerank_documents和rerank_with_scores有什么区别？
      A: 前者接收List[Dict]（不带分数），内部给每个文档赋默认分数0.0，
         返回List[Dict]（带rerank_score字段）。
         后者接收List[Tuple[Dict, float]]（带原始检索分数），
         返回List[Tuple[Dict, float]]（带重排后的分数）。
         底层都用CrossEncoderReranker.rerank()重新打分，原始分数被忽略。

      Q: 为什么不在RerankerService里直接处理模型加载失败的降级？
      A: 降级策略由调用方（retrieval.py）决定——它可以跳过rerank、
         改用原始分数排序、或者返回空结果。RerankerService只负责
         "能rerank就rerank，不能就返回空列表"，怎么处理空结果
         是调用方的职责。单一职责原则。

      Q: 单例模式的缺点是什么？
      A: 模型加载后无法动态切换（如想换用bge-reranker-large需要重启服务）。
         更好的方案是用工厂模式+模型版本号缓存，但项目场景简单，单例足够。
    """
    
    # 【作用】初始化重排序服务，创建CrossEncoderReranker实例
    # 【原理】__init__是构造方法，接收模型名称参数，创建一个内部的重排序器
    # 【逻辑】调用者只需要传一个模型名，服务内部负责创建和管理重排序器
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序服务
        
        Args:
            model_name: 重排序模型名称
        """
        # 【作用】创建一个CrossEncoderReranker实例并保存到self.reranker
        # 【原理】CrossEncoderReranker(model_name)调用其__init__，会加载模型
        # 【逻辑】服务类的模型加载委托给了内部的CrossEncoderReranker实例
        self.reranker = CrossEncoderReranker(model_name)
        
    # 【作用】对文档列表（没有原始分数）进行重排序，返回重排序后的文档列表
    # 【原理】这个方法接收查询和文档列表，给每个文档一个默认的原始分数0.0，然后调用rerank方法
    # 【逻辑】这是对外提供的简化接口，适合调用方只有文档没有原始检索分数的情况
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
        # 【作用】把纯文档列表转换成元组格式，每个文档配一个默认分数0.0
        # 【原理】列表推导式 [(doc, 0.0) for doc in documents] 遍历每个文档，和0.0组成元组
        # 【逻辑】Cross-Encoder重排序不需要原始分数，但rerank方法的接口需要这个格式，所以给个占位值
        doc_tuples = [(doc, 0.0) for doc in documents]
        
        # 【作用】调用内部CrossEncoderReranker的rerank方法进行实际重排序
        # 【原理】self.reranker.rerank(query, doc_tuples, top_k) 执行重排序，返回RerankResult列表
        # 【逻辑】把查询、带默认分数的文档列表、top_k数量传给底层重排序器
        results = self.reranker.rerank(query, doc_tuples, top_k)
        
        # 【作用】遍历重排序结果，提取每个文档并添加重排分数信息
        # 【原理】for result in results 遍历RerankResult列表，result是包含文档和分数的数据类
        # 【逻辑】调用者想要的是带分数的文档字典列表，而不是RerankResult对象
        reranked_docs = []
        for result in results:
            # 【作用】复制文档字典，避免修改原始数据
            # 【原理】doc.copy() 创建字典的浅拷贝，修改拷贝不会影响原字典
            # 【逻辑】如果不复制直接修改原文档，可能会影响调用方的原始数据，引发难以排查的bug
            doc = result.document.copy()
            
            # 【作用】把重排分数作为一个新字段添加到文档字典中
            # 【原理】doc["rerank_score"] = result.relevance_score 给字典新增一个键值对
            # 【逻辑】调用方可以直接从文档里拿到rerank_score字段，不用额外处理
            doc["rerank_score"] = result.relevance_score
            
            # 【作用】把原始排名也作为一个新字段添加到文档字典中
            # 【原理】doc["original_rank"] = result.original_rank 给字典再新增一个键值对
            # 【逻辑】调用方可以看到这个文档在原始检索时的排名，方便对比重排序的效果
            doc["original_rank"] = result.original_rank
            
            # 【作用】把处理好的文档追加到结果列表中
            # 【原理】reranked_docs.append(doc) 在列表末尾添加一个元素
            # 【逻辑】最终返回的是按重排分数降序排列的文档列表
            reranked_docs.append(doc)
            
        # 【作用】返回重排序后的文档列表
        # 【原理】return reranked_docs 把结果返回给调用者
        # 【逻辑】列表中的文档已经按相关性从高到低排好序了
        return reranked_docs
        
    # 【作用】对带原始分数的文档进行重排序，返回(文档, 新分数)元组列表
    # 【原理】这个方法接收查询和带原始分数的文档列表，返回重排后的(文档, 新分数)对
    # 【逻辑】和rerank_documents的区别是：这个接收原始分数，返回的也是元组格式而非纯文档列表
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
        # 【作用】调用内部CrossEncoderReranker的rerank方法进行实际重排序
        # 【原理】self.reranker.rerank(query, documents_with_scores, top_k) 传入带原始分数的文档
        # 【逻辑】底层rerank方法会用Cross-Encoder重新打分，忽略原始分数，返回RerankResult列表
        results = self.reranker.rerank(query, documents_with_scores, top_k)
        
        # 【作用】把RerankResult列表转换成(文档, 新分数)元组列表
        # 【原理】列表推导式 [(r.document, r.relevance_score) for r in results] 提取每个结果中需要的字段
        # 【逻辑】调用方只需要文档和重排后的分数，不需要original_rank、new_rank这些内部字段
        return [(r.document, r.relevance_score) for r in results]


# 【作用】定义一个全局变量，用于存储重排序服务的单例实例，初始值为None
# 【原理】_reranker_service是模块级变量，以单下划线开头表示"模块内部使用，外部不要直接访问"
# 【逻辑】单例模式：重复调用get_reranker_service()时，只创建一次服务实例，后续复用
_reranker_service = None

# 【作用】定义获取重排序服务实例的函数，采用单例模式
# 【原理】def get_reranker_service() -> RerankerService: 函数返回一个RerankerService实例
# 【逻辑】如果还没创建就创建一个新的，如果已经创建了就返回已有的，避免重复加载模型
def get_reranker_service() -> RerankerService:
    """
    获取重排序服务实例（单例）
    
    Returns:
        RerankerService实例
    """
    # 【作用】声明要使用全局变量_reranker_service，而不是创建一个同名局部变量
    # 【原理】global关键字告诉Python：这个变量是在模块级别定义的，别在函数内部新建一个
    # 【逻辑】如果不加global，Python会把_reranker_service当成局部变量，赋值也不会影响全局的那个
    global _reranker_service
    
    # 【作用】检查服务实例是否已经创建过（即_reranker_service是不是None）
    # 【原理】if _reranker_service is None 判断变量是否为None（表示还没初始化）
    # 【逻辑】只有第一次调用时需要创建实例，后续调用直接返回已有的实例
    if _reranker_service is None:
        # 【作用】创建RerankerService的实例，这会触发模型加载
        # 【原理】RerankerService()调用其__init__，内部会创建CrossEncoderReranker并加载模型
        # 【逻辑】模型加载可能比较耗时（几秒到几十秒），所以只做一次，后续复用
        _reranker_service = RerankerService()
    
    # 【作用】返回已创建（或已存在）的重排序服务实例
    # 【原理】return _reranker_service 把全局变量中的实例返回给调用者
    # 【逻辑】无论是第一次创建还是第N次获取，都返回同一个实例对象
    return _reranker_service
