# # -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
import warnings
warnings.filterwarnings('ignore')
# 1.读取Digits数据集
x,y = load_digits(return_X_y=True)
print(x)
# 2.标准化数值
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
# 3.数据切分处理
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.8,random_state=42)
# 4.将数据转换为Tensor
import torch as tc
from torch  import nn
x_train = tc.tensor(x_train,dtype=tc.float)
x_test = tc.tensor(x_test,dtype=tc.float)
y_train = tc.tensor(y_train,dtype=tc.long)
y_test = tc.tensor(y_test,dtype=tc.long)
# 5.创建多层感知机模型
import numpy as np
y_dim = len(np.unique(y_train))
model = nn.Sequential(
    nn.Linear(in_features=x_train.shape[1],out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10,out_features=y_dim),
    nn.Softmax()
)
# 6.模型训练，训练200次（可以根据需要调整）
losses = nn.CrossEntropyLoss()
opt = tc.optim.Adam(model.parameters(),lr=0.01)
model.train()
loss_list = []
for i in range(200):
    # 7.初始化梯度
    losses.zero_grad()
    # 8.前向传播
    y_pre = model(x_train)
    # 9.反向传播计算梯度
    # 10.更新模型参数
    # 11.每10次打印损失值和准确率
    loss = losses(y_pre,y_train)
    loss.backward()
    opt.step()
    loss_list.append(loss.item())
    if i % 10 ==0:
        acc = (y_pre.argmax(axis=1) == y_train).float().mean()
        print(f'第{i}次训练，损失为{loss}，准确率为{acc}')
    # 12.计算测试集上的准确率
model.eval()
y_pred = model(x_test)
val_acc = (y_pred.argmax(axis=1) == y_test).float().mean()
print(val_acc)
import matplotlib.pyplot as plt
plt.plot(loss_list,c = 'r',label = 'loss')
plt.legend()
plt.show()