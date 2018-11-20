#这个文件是直接抄tensorflow官方样例的，可以直接运行
#这是基于logistics regression的手写字体识别
#原文地址 http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html



from tensorflow.examples.tutorials.mnist import input_data
print ("loading data...")
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print ("data loaded.")

import tensorflow as tf
x = tf.placeholder("float",[None,28*28])
w = tf.Variable(tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w)+b)# tensorflow 的softmax是以行为单位计算的
y_=tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for i in range(5000): # 1000个epoch的准确率和5000个epoch的准确率差别不大，1个百分点。训练集准确率为92%,测试集准确率也差不多是92%，这是典型的high bias现象，所以应该增加网络复杂度
        batch_xs,batch_ys= mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

print("trainning accuracy = ",sess.run(accuracy,feed_dict={x:mnist.train.images,y_:mnist.train.labels}))
print("testing accuracy = ", sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

# ---------------below for test
# t1=tf.constant([[4,5,6],[1,8,9]],"float")
# t2=tf.constant(2,"float")
# t3= tf.placeholder("float")
# t3=tf.nn.softmax(t1)

# init = tf.initialize_all_variables()
# sess = tf.Session()

# sess.run(init)
# print (sess.run(t3))