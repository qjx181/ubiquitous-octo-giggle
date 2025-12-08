# # -*- coding: utf-8 -*-
#
# 【卷积特征图计算】
# 给定4×4图像：
# TEXT
# ┌────┬────┬────┬────┐
# │ 10 │  5 │  8 │ 12 │
# ├────┼────┼────┼────┤
# │  3 │ 15 │  6 │  9 │
# ├────┼────┼────┼────┤
# │  7 │  2 │ 11 │  4 │
# ├────┼────┼────┼────┤
# │ 14 │  1 │ 13 │  0 │
# └────┴────┴────┴────┘
#
#
#
#
#
# 使用3×3卷积核：
# TEXT
# ┌────┬────┬────┐
# │  1 │  0 │  1 │
# ├────┼────┼────┤
# │  1 │  0 │  1 │
# ├────┼────┼────┤
# │  1 │  0 │  1 │
# └────┴────┴────┘
# 问题要求：计算特征图（2×2），并写出计算过程。
#
print(10*1+5*0+8*1+3*1+15*0+6*1+7*1+2*0+11*1)
print(5*1+8*0+12*1+15*1+6*0+9*1+2*1+11*0+4*1)
print(3*1+15*0+6*1+7*1+2*0+11*1+14*1+1*0+13*1)
print(15*1+6*0+9*1+2*1+11*0+4*1+1*1+13*0+0*1)
# 2.【Softmax损失计算】
# 分类模型输出：
# TEXT
# 类别A: 0.10 | 类别B: 0.25 | 类别C: 0.35 | 类别D: 0.15 | 类别E: 0.15
# 真实类别为 类别C（索引2）。
# 问题要求：计算交叉熵损失。
print(0.10+0.25+0.35+0.15+0.15)
import numpy as np
print(-1*np.log(0.35))
# 3. 【特征图尺寸计算】
# 输入尺寸：800×800
# 卷积核：9×9
# 步长：4
# 填充方式：same
# 问题要求：计算输出特征图尺寸。
padding = 9//2
print(((800+2*padding)-9)//4+1)
#
# 1.模型约束描述
# 1)数据加载与预处理：
# 加载CIFAR-10数据集，该数据集包含60000张32x32的彩色图像，分为10个类别。
# 类别编号0 - airplane（飞机）
# 1.人造机械类代表
# 2.具有规则几何结构（机翼、机身等）
# 类别编号1 - automobile（汽车）
# 1.地面交通工具代表
# 2.与飞机形成"运输工具"对比组
# 3.保留矩形轮廓识别挑战
# 类别编号3 - cat（猫）
# 1.动物类代表
# 2.柔性有机形态（与机械类对比）
# 3.保留毛发纹理识别难度
# 类别编号5 - dog（狗）
# 1.动物类对照样本
# 2.与猫构成"宠物"细分类别
# 3.测试模型对相似生物特征的区分能力
# 选择逻辑：
# 建立「机械vs生物」的二元对比框架
# 包含交通工具（飞机/汽车）和宠物（猫/狗）两个天然分类场景
# 保留四者间的大小比例差异（飞机通常比汽车大、猫狗相对尺寸接近）
# 维持足够的分类难度（特别是猫狗细粒度区分）
# 此组合既能验证模型对刚性结构和柔性形态的识别能力，又能测试其跨尺度特征提取性能，常被用于评估CNN在基础视觉概念理解上的表现。
# 将训练集和测试集的像素值归一化到[0, 1]范围，即将原始像素值除以255
from keras.src.datasets import cifar10
from keras.src.utils import to_categorical
# 从CIFAR-10数据集中精选以下4个类别，构成具有视觉区分度与分类挑战性的组合：
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

train_filter = np.where((y_train == 0)|(y_train == 1)|(y_train == 3)|(y_train == 5))[0]
test_filter = np.where((y_test == 0)|(y_test == 1)|(y_test == 3)|(y_test == 5))[0]

x_train = x_train[train_filter].reshape(-1,32,32,3).astype('float32')/255.0
x_test = x_test[test_filter].reshape(-1,32,32,3).astype('float32')/255.0

y_train = y_train[train_filter]
y_test = y_test[test_filter]

y_dim = len(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 2)模型定义：
#
from keras.src import Model,losses,legacy,layers,Sequential
# 3)模型构建与编译：

# 在主函数中，实例化VGG16类创建一个模型对象。
    # 使用模型创建方法
    # 调用方法打印模型的摘要信息。
    # 使用方法编译模型，指定优化器为Adam，损失函数，评估指标为准确率。

class VGG16(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.con = Sequential([
            layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),
            layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),
            layers.MaxPooling2D()
        ])
        self.fa = Sequential([layers.Flatten()])
        self.fc = Sequential([
            layers.Dense(units=64,activation='relu'),
            layers.Dense(units=64,activation='relu'),
            layers.Dense(units=y_dim,activation='relu')
        ])
    def call(self, x,*args, **kwargs):
        x = self.con(x)
        x = self.fa(x)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    model = VGG16()
    model.build(input_shape = x_train.shape)
    model.compile(
        loss=losses.categorical_crossentropy,
        optimizer='Adam',
        metrics=['acc']
    )
    h = model.fit(x_train,y_train,epochs=3,batch_size=100,validation_data=(x_test,y_test))
        # 4)模型训练：
    # 使用方法训练模型，指定训练集和测试集，训练过程中，模型的性能（准确率和损失）会在验证集上进行评估。
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    acc = h.history['acc']
    val_acc = h.history['val_acc']
    import matplotlib.pyplot as plt
    plt.plot(loss,c = 'r',label = 'loss')
    plt.plot(val_loss,c = 'g', label = 'val_loss')
    plt.legend()
    plt.show()
    plt.plot(acc,c = 'r',label = 'acc')
    plt.plot(val_acc,c = 'g', label = 'val_acc')
    plt.legend()
    plt.show()
    # 5)训练结果可视化：
    # 从训练日志中提取训练集和验证集的准确率。
    # 使用方法绘制训练集和验证集准确率随轮数变化的曲线图，红色表示训练集准确率，绿色表示验证集准确率。
    # 6)模型权重保存：
    model.save_weights('model.weights.h5')
    '''
    当网络结构为正常vgg16时 acc: 0.5371 - loss: 1.0016 - val_acc: 0.5395 - val_loss: 0.9960
    当添加一层卷积层时 acc: 0.3059 - loss: 1.3432 - val_acc: 0.4475 - val_loss: 1.1073
    当添加两层卷积层时 acc: 0.5100 - loss: 1.0238 - val_acc: 0.5508 - val_loss: 1.0109
    '''
    # 训练完成后，使用方法将模型的权重保存到文件中，以便后续加载和推理。
    # 模型调优，考题中要求使用体现不同优化方案的要求，比如超参数方面的，
    # 比如说分类精度低的数据集的增强；比如说调整网络结构，比如说调整优化器学习率等；
    # 不能只是增加epoch大小，或者调整batch_size就可以，
    # 必须采用两种或者两种以上的调优方法进行调优，最终结论要给出综合各种方法调优得出的综合结论
#
