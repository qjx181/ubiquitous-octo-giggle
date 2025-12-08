# -*- coding: utf-8 -*-
# 项目2：基于GRU模型的波士顿房价时间序列预测
# 一、项目背景
# 波士顿房价数据集是一个经典的机器学习数据集，包含了波士顿地区房屋的各种特征以及对应的房价中位数。这个数据集常被用来作为回归任务的基准数据集，帮助研究者和开发者评估和优化他们的机器学习模型。近年来，随着深度学习技术的快速发展，越来越多的研究者开始尝试使用深度学习模型，如循环神经网络（RNN）及其变体，来处理这类时间序列预测问题。
# 二、项目目标
import torch
from sklearn.datasets import load_boston
from torch import nn
import torch as tc
import warnings
warnings.filterwarnings('ignore')
# 三、功能需求
# 1.数据集加载与预处理：
# 必须使用波士顿房价数据集。
x,y = load_boston(return_X_y=True)

# 数据集的特征和目标变量都需要进行归一化处理。
from sklearn.preprocessing import MinMaxScaler
x = MinMaxScaler().fit_transform(x)
y = MinMaxScaler().fit_transform(y.reshape(-1,1))
# 需要创建时间序列数据，其中每个样本包含连续7个时间点的特征数据，并对应一个时间点的目标变量。
c = 7
x_ = []
y_ = []
for i in range(len(x)-c):
    x_.append(x[i:i+c])
    y_.append(y[i+c])
x = tc.tensor(x_,dtype=tc.float)
y = tc.tensor(y_,dtype=tc.float)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=False,random_state=42)
# 2.模型构建：
# 模型必须是一个基于GRU（门控循环单元）的深度学习模型。
class GRU(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.g1 = nn.GRU(input_size=x_train.shape[2],hidden_size=10,batch_first=True)
        self.g2 = nn.GRU(input_size=10,hidden_size=10,batch_first=True)
        self.lin = nn.Linear(in_features=10,out_features=1)
    def forward(self,x):
        x,_ = self.g1(x)
        x,_ = self.g2(x)
        x  = self.lin(x[:,-1,:])
        return x
# 3.训练过程：
if __name__ == '__main__':
    model = GRU()
    losses = nn.MSELoss()
    opt = tc.optim.Adam(model.parameters(),lr=0.01)
    # 使用均方误差（MSE）作为损失函数。
    # 使用Adam优化器对模型参数进行优化。

    # 训练模型，记录训练过程中的损失和验证损失，以便进行模型调优。
    loss_list = []
    model.train()
    for i in range(100):
        losses.zero_grad()
        y_pre = model(x_train)
        loss = losses(y_pre,y_train)
        if i % 10 == 0:
            loss_list.append(loss.item())
            print(loss.item())
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        import matplotlib.pyplot as plt
        plt.plot(y_pred,c = 'r',label = 'pred')
        plt.plot(y_test,c = 'g',label = 'test')
        plt.legend()
        plt.show()
'''
当学习率为0.01 时 最终损失为0.09380461275577545
当学习率为0.01 时 最终损失为1.2485133409500122
当学习率为0.01 时 最终损失为9.613720893859863
当学习率为0.01是模型最优
'''
    # 4.评估与可视化：
    # 使用测试集评估模型的预测性能。
    # 必须提供预测结果与实际房价的可视化对比。
    # 可视化图表中应包含图例，以便区分预测结果和实际房价。
    # 针对预测结果与实际房价的可视化对比，以注释形式总结预测模型可
    # 能存在的问题。
# 5.模型优化：
# 尝试不同的网络结构（如改交GRU层的数量、单元数），观察对模型性能的影响或者调整超参数（如学习率、批次大小、Dropout率），通过找到最优参数组合。
# 6.代码可读性与可维护性：
# 代码应具有良好的可读性和可维护性。
# 注释应清晰明了，以便他人理解代码的功能和逻辑。
# 7.其他约束：
# 不得更改数据集、模型类型、损失函数、优化器或学习率等关键配置，除非有明确的优化或改进需求。
# 必须使用PyTorch框架进行模型构建和训练。
# 可视化图表应使用matplotlib库进行绘制。
# 模型调优，考题中要求使用体现不同优化方案的要求，比如超参数方面的，比如说分类