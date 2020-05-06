from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np
from skimage import util
#
# image_name= "C:/Users/sunyihuan/Desktop/x6/20200227_010722979.jpg"
# image=Image.open(image_name)
# image=image.resize((400, 300))
# image.save("C:/Users/sunyihuan/Desktop/x6/20200227_010722979_400.jpg")

# image_path = "C:/Users/sunyihuan/Desktop/85_new_cam/20191216_062713704.jpg"
# image = Image.open(image_path)
# image = ImageEnhance.Contrast(image)  # 对比度增强
# image = image.enhance(random.uniform(0.6, 1.2))  # 增强系数[0.6, 1.2]
# print(np.array(image))
# image = util.random_noise(np.array(image), mode="gaussian")  # 加入高斯噪声,输出值为[0,1],需乘以255
# image = image * 255
#
# print(np.array(image).astype(int))

import tensorflow as tf

pb_file = "E:/ckpt_dirs/Food_detection/local/20191216/yolo_model.pb"

with tf.Session() as sess:
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')

input_array = ['define_input/input_data', 'define_input/training']
output = ["define_loss/pred_sbbox/concat_2", "define_loss/pred_mbbox/concat_2",
          "define_loss/pred_lbbox/concat_2", "define_loss/layer_classes"]

# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
input_shape={'define_input/input_data':[None,416,416,3],'define_input/training':[None]}
converter = tf.lite.TFLiteConverter.from_session(sess, input_array, output)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# import tensorflow as tf
#
# # 声明两个变量
# v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
# v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
# init_op = tf.global_variables_initializer()  # 初始化全部变量
# saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
# with tf.Session() as sess:
#     sess.run(init_op)
#     print("v1:", sess.run(v1))  # 打印v1、v2的值一会读取之后对比
#     print("v2:", sess.run(v2))
#     saver_path = saver.save(sess, "save/model.ckpt")
