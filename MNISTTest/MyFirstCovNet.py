# A simple convolution neural network using MNIST
# 准确地来说是全连接网络，最后准确率在99%
# Author: King of BD
# 2018-11-16

from tensorflow.examples.tutorials.mnist import input_data
print ("loading data...")
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print ("data loaded.")

import tensorflow as tf

sess = tf.InteractiveSession()

def w_generate(shape):
    tmp = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(tmp)

def b_generate(shape):
    tmp = tf.constant(0.1,shape=shape)
    return tf.Variable(tmp)

def cov_2d(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1,],"SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],"SAME")

x = tf.placeholder("float",[None,28*28])
y_t = tf.placeholder("float",[None,10])

x_r = tf.reshape(x,[-1,28,28,1]) # don't use  x=reshape(x,[]), the input and output of reshape cannot be same, image size is 28*28, one channel

w_covn1 = w_generate([5,5,1,32])
b_covn1 = b_generate([32])

layer1_out = tf.nn.relu(cov_2d(x_r,w_covn1)+b_covn1)
layer1_pool_put = max_pool_2x2(layer1_out) #14*14*32

w_conv2 = w_generate([5,5,32,64])
b_conv2 = b_generate([64])

layer2_out = tf.nn.relu(cov_2d(layer1_pool_put,w_conv2)+b_conv2)
layer2_pooling_out = max_pool_2x2(layer2_out) #7*7*64

w_conv3 = w_generate([7,7,64,1024])
b_conv3 = b_generate([1024])

layer3_out = tf.nn.relu(tf.nn.conv2d(layer2_pooling_out,w_conv3,[1,1,1,1],"VALID")+b_conv3) # 1*1*1024
keep_prob = tf.placeholder("float")
layer3_prob_out = tf.nn.dropout(layer3_out,keep_prob)

w_conv4 = w_generate([1,1,1024,10]) # 1x1 convolution, instead of full connection layer
b_conv4 = b_generate([10])

y_p = tf.nn.softmax(cov_2d(layer3_prob_out,w_conv4)+b_conv4)
y_p_r = tf.reshape(y_p,[-1,10])#remenber to reshape before calculating loss

cross_entropy = -tf.reduce_mean(y_t*tf.log(y_p_r))

correct_predict = tf.equal(tf.argmax(y_p_r,1),tf.argmax(y_t,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,"float")) #this two line aim to get accuracy rate

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0],y_t:batch[1],keep_prob:0.5})
    if i%100==0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_t:batch[1],keep_prob:1})
        print ("Step %d,Trainning accuracy = %g"%(i,train_accuracy))

test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images,y_t:mnist.test.labels,keep_prob:1})
print ("Testing accuracy = %g"%(test_accuracy))
