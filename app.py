# -*- coding: utf-8 -*-
# 2.Web应用的搭建功能性约束
from flask import Flask,request,render_template,url_for
from model import VGG16,x_test,x_train
model  = VGG16()
model.build(input_shape = x_train.shape)
model.save_weights('model.weights.h5')

from keras.src.utils import img_to_array,load_img
def read_image(file):
    file = load_img(file,target_size=x_train.shape)
    file = img_to_array(file)
    file = file.reshape(-1,32,32,3).astype('float32')/255.0
    return file

app = Flask(__name__)
@app.route('/')
def demo():
    return render_template('home.html')
@app.route('/predict',methods = ['POST'])
def fun1():
    try:
        if request.method == 'POST':
            file = request.files['file']
            path = 'static/'+file.filename
            file.save(path)
            img = read_image(path)
            y_pre = model.predict(img).argmax()
            dict1 = {0:1,1:1,2:3,3:5}
            return render_template('predict.html',uesr_image = path,product = dict1[y_pre])
    except Exception as e:
        print(e)
# 1)首页展示：
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=6008,debug=True)
# 当用户访问应用程序的根URL（/）时，他们会看到一个名为home.html的HTML页面。这个页面通常包含关于应用程序的信息，以及一个用于上传图像的表单。
# 2)图像上传与保存：
# 在home.html页面上，用户可以通过一个表单选择并上传图像文件。
# 当用户提交表单时，图像文件会被发送到服务器的/predict端点。
# 服务器接收图像文件，并将其保存到static/images目录下，文件名保持不变。
# 3)图像预处理：
# 服务器使用Keras的load_img和img_to_array函数来加载和转换上传的图像。
# 图像被调整为32x32像素的大小，并被转换为一个NumPy数组。
# 数组被重新塑形为模型所需的输入格式（1x32x32x3），并且像素值被归一化到0到1之间。
# 4)模型预测：
# 预处理后的图像被传递给一个预先训练好的VGG16模型（通过cifar10_vgg16.VGG16类实例化）。
# 模型对图像进行分类预测，并返回一个包含10个概率值的数组（对应CIFAR-10数据集的10个类别）。
# 使用np.argmax函数找到概率最高的类别索引。
# 5)模型优化：
# 选择一种优化方法对模型进行进一步优化（如调整学习率、使用正则化、剪枝等），并分析优化后的结果。
# 6)结果显示：
# 预测结果（类别名称）以及上传的图像一起显示在predict.html模板中。
# 类别名称是通过一个字典将类别索引映射到中文名称来获得的。
# 7)错误处理：
# 如果在图像上传、预处理或预测过程中发生错误，应用程序会捕获异常并返回错误信息。
# 错误信息以字符串形式返回，并显示在用户的浏览器中。
# 8)服务器配置：
# 应用程序在调试模式下运行，这意味着任何代码更改都会立即生效，而无需重新启动服务器。
# 服务器监听在所有网络接口上的6008端口，允许来自任何IP地址的连接。

# 4)文件上传：
# 当用户选择文件并点击“Upload”按钮时，浏览器将文件作为多部分表单数据按正确的加密方式发送到服务器。
# 5)服务器端处理：
# 服务器端的/predict端点需要配置为接收这个文件，并可能执行一些处理（例如，图像识别、分类等），然后返回预测结果。
# 5.数据字段以及数据情况说明
# 1)图像数据
# cifar10_labels = {
#     0: 'airplane（飞机）',
#     1: 'automobile（汽车）',
#     2: 'bird（鸟）',
#     3: 'cat（猫）',
#     4: 'deer（鹿）',
#     5: 'dog（狗）',
#     6: 'frog（青蛙）',
#     7: 'horse（马）',
#     8: 'ship（船）',
#     9: 'truck（卡车）'
# }
#
#
