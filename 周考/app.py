# -*- coding: utf-8 -*-
# 1)用户交互：
# 提供一个简单的用户界面，允许用户上传图片。
# 在用户上传图片后，显示预测结果。
# 2)图片处理：
# 接收用户上传的图片文件。
# 将图片保存到服务器上的指定目录。
# 读取图片文件，并将其转换为模型可以处理的格式。
# 3)模型预测：
# 加载预先训练好的深度学习模型。
# 使用模型对处理后的图片进行预测。
# 将预测结果转换为人类可读的格式。
# 4)结果展示：
# 将预测结果展示给用户，包括预测的类别和上传的图片。
# 5)错误处理：
# 如果上传的文件不是图片，或者文件读取出现问题，需要给用户一个明确的错误提示。
# 6)服务器配置：
# 配置Flask应用以在指定的主机和端口上运行。
# 开启调试模式以便于开发过程中的错误追踪。
# 7)安全性：
# 确保上传的文件被保存在安全的目录下，并且只能通过Web应用访问。
# 防止恶意文件上传和执行。
# 8)可扩展性：
# 代码结构应该清晰，以便于未来添加更多的功能或修改现有的功能。
# 9)性能：
# 确保应用能够处理多个并发请求。
# 优化图片处理和模型预测的速度。
# 10)兼容性：
# 确保应用在不同的浏览器和操作系统上都能正常工作。
# 3.前端上传图像页功能约束
# 1)文件上传功能：
# 用户应能够选择一个文件并通过表单提交。
# 提交的文件必须是图片格式（如.jpg, .png, .gif等）。
# 文件大小应有限制，例如不超过5MB。
# 2)表单提交行为：
# 当用户点击“Upload”按钮时，表单数据应通过POST方法发送到服务器的"/predict"路径。
# 表单数据应以"multipart/form-data"格式编码，以便可以上传文件。
#
# 4)错误处理和反馈：
# 如果文件类型不正确或文件大小超过限制，用户应收到适当的错误消息。
# 如果服务器处理请求时发生错误，用户应得到相应的反馈。
# 4.前端预测页功能约束

# 5)安全性和隐私：
# 如果用户图像包含敏感信息，服务器应确保不会泄露这些信息。
# 用户的预测结果和图像数据应得到妥善处理，遵守相关的数据保护法规。
# 6)错误处理和反馈：
# 如果服务器无法处理请求或图像无法加载，用户应收到适当的错误消息。
# 表单提交后，如果发生错误，用户应被重定向到一个错误页面或收到一个错误提示。
# 5.数据字段以及数据情况说明
#
#
from flask import Flask,request,render_template
app = Flask(__name__)
from model import AlexNet,x_test,x_train
model = AlexNet()
model.build(x_train.shape)
model.load_weights('model.weights.h5')

from keras.src.utils import load_img,img_to_array
def read_image(file):
    file = load_img(file,target_size=x_train.shape)
    file = img_to_array(file)
    file = file.reshape(-1,32,32,3).astype('float32')/255.0
    return file

@app.route('/')
def demo():
    return render_template('home.html')
@app.route('/predict',methods = ['POST'])
def fun1():
    try:
        file = request.files['file']
        path = 'static/'+file.filename
        file.save(path)
        image = read_image(path)
        y_pre = model.predict(image.reshape(-1,32,32,3)).argmax()
        cifar10_labels = {
        0: 'airplane（飞机）',
        1: 'automobile（汽车）',
        2: 'bird（鸟）',
        3: 'cat（猫）',
        4: 'deer（鹿）',
        5: 'dog（狗）',
        6: 'frog（青蛙）',
        7: 'horse（马）',
        8: 'ship（船）',
        9: 'truck（卡车）'
    }
        img_new = cifar10_labels[y_pre]
        return render_template('predict.html',uesr_image = path,product = img_new)
    except Exception as e :
        print(e)
    return render_template('predict.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=6008,debug=True)




