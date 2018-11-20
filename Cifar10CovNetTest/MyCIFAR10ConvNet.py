# Author: King of BD
# 使用了tensorflow官方提供的读cifar10的接口，即cifar10.py和cifar10_input.py，
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


batch_size = 50
cifar10.maybe_download_and_extract()#下载数据集
#训练集
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
#cifar10_input类中带的distorted_inputs()函数可以产生训练需要的数据，包括特征和label，返回封装好的tensor，每次执行都会生成一个batch_size大小的数据。
#测试集
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir=data_dir, batch_size=10000)




def w_generate(shape,stddev=0.1,w=0):
    tmp = tf.truncated_normal(shape,stddev=stddev)
    return tf.Variable(tmp)

def b_generate(shape,value=0.1):
    tmp = tf.constant(value=value,shape=shape)
    return tf.Variable(tmp)

def cov_2d(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1,],"SAME")

def max_pool_2x2(x,ksize = [1,2,2,1],strides=[1,2,2,1]):
    return tf.nn.max_pool(x,ksize,strides,"SAME")


#data preparing
x = tf.placeholder("float",[None,24,24,3])
y = tf.placeholder("int32",[None])
y_t = tf.to_float(tf.one_hot(y,10,1,0)) #y_t means y true

print("y shape = ",tf.shape(y))

# w_covn1 = w_generate([5,5,3,32])
# b_covn1 = b_generate([32])

# layer1_out = tf.nn.relu(cov_2d(x,w_covn1)+b_covn1)
# layer1_pool_put = max_pool_2x2(layer1_out) #12*12*32

# w_conv2 = w_generate([5,5,32,64])
# b_conv2 = b_generate([64])

# layer2_out = tf.nn.relu(cov_2d(layer1_pool_put,w_conv2)+b_conv2)
# layer2_pooling_out = max_pool_2x2(layer2_out) #6*6*64



# layer2_pooling_out_flat = tf.reshape(layer2_pooling_out,[-1,6*6*64])
# fc_w1 = tf.Variable(tf.truncated_normal([6*6*64,1024],stddev=0.1))
# fc_b1 = tf.Variable(tf.constant(0.1,shape=[1024]))

# fc_out1 = tf.nn.relu(tf.matmul(layer2_pooling_out_flat,fc_w1)+fc_b1)

# keep_prob = tf.placeholder("float")
# fc1_prob_out = tf.nn.dropout(fc_out1,keep_prob)

# fc_w2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
# fc_b2 = tf.Variable(tf.constant(0.1,shape=[10]))

# fc_out2 = tf.nn.softmax(tf.matmul(fc1_prob_out,fc_w2)+fc_b2)
# y_o = tf.reshape(fc_out2,[-1,10])

#卷积层1
w1 = w_generate([5,5,3,64],stddev=5e-2)
b1 = b_generate([64])

layer1_c = tf.nn.relu(tf.add(cov_2d(x,w1),b1))
layer1_p = max_pool_2x2(layer1_c,ksize=[1,3,3,1],strides=[1,2,2,1])
layer1_n = tf.nn.lrn(layer1_p,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

#卷积层2
w2 = w_generate([5,5,64,64],stddev=5e-2)
b2 = b_generate([64])

layer2_c = tf.nn.relu(tf.add(cov_2d(layer1_n,w2),b2))
layer2_n = tf.nn.lrn(layer2_c,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
layer2_p = max_pool_2x2(layer2_n,ksize=[1,3,3,1])
print (layer2_p.shape)

#卷积层3
w2_1 = w_generate([5,5,64,64],stddev=5e-2)
b2_1 = b_generate([64])

layer3_c = tf.nn.relu(tf.add(cov_2d(layer2_p,w2_1),b2_1))
layer3_n = tf.nn.lrn(layer3_c,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
# layer3_p = max_pool_2x2(layer3_n,ksize=[1,3,3,1])
print (layer3_n.shape)

# 全连接层1
w3 = w_generate([6*6*64,384],stddev=0.04)
b3 = b_generate([384])
reshape = tf.reshape(layer3_n,[-1,6*6*64])
fc1 = tf.nn.relu(tf.add(tf.matmul(reshape,w3),b3))

#全连接层2
w4 = w_generate([384,192],stddev=0.04)
b4 = b_generate([192])
fc2 = tf.nn.relu(tf.add(tf.matmul(fc1,w4),b4))

keep_prob = tf.placeholder("float")
fc2_d = tf.nn.dropout(fc2,keep_prob)

#输出层
w5 = w_generate([192,10],stddev=1/199.0)
b5 = b_generate([10])

# y_o = tf.nn.softmax(tf.nn.relu(tf.matmul(fc2_d,w5)+b5))
# cross_entropy = -tf.reduce_mean(y_t*tf.log(y_o)) #y_O means output of model
#之前使用上面自己写的方式计算交叉熵，结果在训练十几万次后会出现准确率突然骤降为10%左右的情况。然后改成使用tensorflow自带的交叉熵函数计算，就没问题了


y_o = tf.nn.relu(tf.add(tf.matmul(fc2_d,w5),b5))#网络最终输出结果
cross_entropy_tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=y_o)#计算交叉熵第一步
cross_entropy = tf.reduce_mean(cross_entropy_tmp)#计算交叉熵

correct_predict = tf.equal(tf.argmax(y_o,1),tf.argmax(y_t,1))#计算正确率第一步  
accuracy = tf.reduce_mean(tf.cast(correct_predict,"float")) #计算正确率
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#训练方法设定

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()#开启多线程，预处理图像

saver = tf.train.Saver(max_to_keep=2)#保存网络，这个函数作用是  保存网络  的配置信息（不是保存网络的  配置信息 ）， 保存网络  在这里是个名词

train_steps = 20000


log_file = open("log.txt","w")
line = "Start training at "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n"
log_file.write(line)
log_file.flush()
loss_array = np.zeros(train_steps)
count_tmp = 0


start_time = time.time()
for i in range(train_steps):#200000
#   batch_xs, batch_ys = dr.next_train_data(50)
    batch_xs, batch_ys = sess.run([images_train, labels_train])
    train_step.run(feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})

    if i%100==0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
        loss = cross_entropy.eval(feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
        loss_array[count_tmp] = loss
        print ("Training %.2f %%,Trainning accuracy = %f%%, loss = %.4f"%(100*i/train_steps,100*train_accuracy,loss))
        line = "Training %.2f %%,Step = %g Trainning accuracy = %f%%, loss = %.4f \n"%(100*i/train_steps,i,100*train_accuracy,loss)
        log_file.write(line)
        log_file.flush()
        count_tmp+=1
        if train_accuracy > 0.92:
                saver.save(sess,'Cifar10CovNet_model',global_step=i,write_meta_graph=True)




line = "End training at "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n"
log_file.write(line)
log_file.flush()


tmp = np.linspace(1,count_tmp,count_tmp)
plt.plot(tmp,loss_array[0:count_tmp])
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("Training loss")
plt.savefig("loss.jpg")
plt.show()

duration = time.time() - start_time #运行时间
batch_xs, batch_ys = sess.run([images_test, labels_test])
test_accuracy = accuracy.eval(feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
print ("Testing accuracy = %g,training time = %g sec"%(test_accuracy,duration))

line = "Testing accuracy = %g,training time = %g sec "%(test_accuracy,duration)
log_file.write(line)
log_file.flush()
log_file.close()