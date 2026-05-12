from FlagEmbedding import BGEM3FlagModel
import torch

device = torch.device("cuda")
print("当前使用设备:", device)

# 现在可以正常加载模型了！
model = BGEM3FlagModel(r'C:\Users\qjx\.cache\modelscope\hub\models\BAAI\bge-m3', use_fp16=True, device=device)
print("模型所在设备:", next(model.model.parameters()).device)