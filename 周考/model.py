# # -*- coding: utf-8 -*-
# 1.模型约束描述

# 1)数据加载与预处理：
# 加载CIFAR-10数据集。
# 模型的卷积部分由5个卷积层和3个最大池化层交替组成，用于提取图像特征。
# 卷积层之后是一个扁平化层，用于将多维的特征图转换为一维的向量。
# 最后是全连接层部分，包含两个64单元的全连接层和一个10单元的输出层，输出层使用激活函数进行多分类。
# 3)模型编译：
# 使用优化器。
# 损失函数为交叉熵损失函数，适用于多分类问题且标签为整数的情况。
# 评估指标为准确率。
# 4)模型训练：数据集划分为训练集、验证集和测试集；评估指标为精度（精确率或召回率或F1） 和损失值；训练过程中，可视化展示训练集、验证集的精度、1oss的变化曲线，并简要分析模型的训练效果
# （分析的内容以注释的形式总结出来）；将训练好的模型权重保存到model.n5文件中
# 5)模型评估：在测试集上使用保存下来的模型文件评估分类模型的性能，包括准确率或混浠矩阵等指标 
# 6)模型优化：选择一种优化方法对模型进行进一步优化（如调整学习率或更改优化器或调整神经元数量），并分析优化后的结果使用测试集数据评估模型的性能。
# 2.Web应用的搭建功能性约束
from keras.src.datasets import cifar10
import numpy as np
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# 将训练集和测试集的图像数据归一化。
print(x_train.shape)
x_train = x_train.reshape(-1,32,32,3).astype('float32')/255.0
x_test = x_test.reshape(-1,32,32,3).astype('float32')/255.0
y_dim = len(np.unique(y_train))
# 2)模型架构：
from keras.src import Model,losses,legacy,layers,Sequential
# 模型是一个自定义的AlexNet类。
class AlexNet(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.con = Sequential([
            layers.Conv2D(filters=6,kernel_size=16,activation='relu',padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=16,kernel_size=16,activation='relu',padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=24,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=24,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same'),
            layers.MaxPooling2D()
        ])
        self.fa = Sequential([layers.Flatten()])
        self.fc = Sequential([
            layers.Dense(units=120,activation='relu'),
            layers.Dense(units=84,activation='relu'),
            layers.Dense(units=y_dim,activation='softmax'),
        ])
    def call(self,inputs, *args, **kwargs):
        x = self.con(inputs)
        x = self.fa(x)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    model = AlexNet()
    model.build(input_shape = (None,32,32,3))
    model.compile(
        loss=losses.sparse_categorical_crossentropy,
        optimizer='Adam',
        metrics=['acc']
    )
    h = model.fit(x_train,y_train,batch_size=100,validation_data=(x_test,y_test),epochs=3)
    loss = h.history['loss']
    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.show()
    model.save_weights('model.weights.h5')
    model.load_weights('model.weights.h5')

