# # -*- coding: utf-8 -*-
# 数据归一化强制要求：
import torch
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')
import torch as tc
# 必须对特征数据（data）和目标值（target）进行归一化处理，确保全部数据落在 [0, 1] 范围内。
x,y = load_boston(return_X_y=True)
from sklearn.preprocessing import MinMaxScaler
x = MinMaxScaler().fit_transform(x)
print(x.shape)
y = MinMaxScaler().fit_transform(y.reshape(-1,1))
# 时序窗口规则：
# 时间窗口长度固定为 7 天（c=7），禁止调整窗口大小或采用动态窗口策略。
c = 7
x_ = []
y_ = []
# 输入数据 x 为连续 7 天的特征数据，目标标签 y 对应第 8 天的房价。
# 数据集划分要求
# 必须使用 sklearn.model_selection.train_test_split 划分数据集，且禁止随机打乱（shuffle=False），以保留时间顺序。
# 测试集必须用于建模未来数据的验证（即测试集的时间在训练集之后），确保时序依赖性不被破坏。
for i in range(len(x)-c):
    x_.append(x[i:i+c])
    y_.append(y[i+c])
x = tc.tensor(x_,dtype=tc.float)
y = tc.tensor(y_,dtype=tc.float)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=False)
x_train  = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test  = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])
# 模型设计约束：
from torch import nn
# 必须使用 PyTorch 的单层线性回归模型（torch.nn.Linear），
model = nn.Sequential(
    nn.Linear(in_features=x_train.shape[1],out_features=10),
    nn.Linear(in_features=10,out_features=1)
)
# 输入维度为 时间窗口长度 × 特征数（展平后的一维数据），输出维度为 1（房价）。
# 禁止使用非线性激活函数（如 ReLU）或其他网络层（如隐藏层、循环层）。
# 优化与损失计算：
losses = nn.MSELoss()
# 损失函数必须为均方误差（MSE，通过 torch.nn.MSELoss 计算）。
# 优化器必须为 Adam（torch.optim.Adam），学习率固定为 0.001，禁止修改或替换（如 SGD/RMSProp）。
opt = tc.optim.Adam(model.parameters(),lr=0.001)
# 训练流程管理：
# 模型必须迭代训练 1000 次，每 10 次输出一次训练损失。
# 必须使用 zero_grad() 清除历史梯度，通过 loss.backward() 计算梯度，
# 并通过 step() 更新模型参数。
loss_list = []
model.train()
for i in range(1000):
    model.zero_grad()
    y_pre = model(x_train)
    loss = losses(y_pre,y_train)
    loss_list.append(loss.item())
    if i % 10 ==0:
        print(loss.item())
    loss.backward()
    opt.step()
# 4.评估与可观测性约束
# 评估模式限制：
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    print(y_pred)
# 模型在测试时必须处于评估模式（model.eval()），并禁用梯度计算（torch.no_grad()）。
# 可视化规范：
import matplotlib.pyplot as plt
plt.plot(y_pred,c = 'g',label = 'pred')
plt.plot(y_test,c = 'r',label = ' text')
plt.legend()
plt.show()
# 必须使用 matplotlib 绘制对比图，真实值以红色曲线（'r'）表示，预测值以绿色曲线（'g'）表示。
