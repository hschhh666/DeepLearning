# Author: King of BD
# 使用了tensorflow官方提供的读cifar10的接口，即cifar10.py和cifar10_input.py，
# 读取已保存的网络模型及参数
# 但是原来的网络没有手动给定每个节点的名称，所以需要从tensorflow给每个节点的默认名称中找到需要的节点。
# github地址https://github.com/tensorflow/models.git ，models/tutorials/image/cifar10/

import tensorflow as tf
import numpy as np
import cifar10
import cifar10_input
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import datetime
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'


batch_size = 500
cifar10.maybe_download_and_extract()#下载数据集
#训练集
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
#cifar10_input类中带的distorted_inputs()函数可以产生训练需要的数据，包括特征和label，返回封装好的tensor，每次执行都会生成一个batch_size大小的数据。
#测试集
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir=data_dir, batch_size=10000)

ckpt = tf.train.get_checkpoint_state('./')
print(ckpt)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

sess = tf.InteractiveSession()
tf.train.start_queue_runners()#开启多线程，预处理图像

saver.restore(sess,ckpt.model_checkpoint_path)
gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量

tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]

name_file = open('name.txt','w')

for i in tensor_name_list:
    name_file.write(i+"\n")
name_file.flush()
name_file.close()

# writer = tf.summary.FileWriter("./", sess.graph)
# writer.close()


x = gragh.get_tensor_by_name('Placeholder:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
y = gragh.get_tensor_by_name('Placeholder_1:0')# 获取输出变量
keep_prob = gragh.get_tensor_by_name('Placeholder_2:0')# 获取dropout的保留参数
accuracy = gragh.get_tensor_by_name('Mean_1:0')# 获取dropout的保留参数



batch_xs, batch_ys = sess.run([images_test, labels_test])
test_accuracy = accuracy.eval(feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})

# batch_xs, batch_ys = sess.run([images_test, labels_test])
# test_accuracy = accuracy.eval(feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
print ("Testing accuracy = %g"%(test_accuracy))


