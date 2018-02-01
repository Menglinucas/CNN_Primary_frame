# 参考极客学院
# 加载库
import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据集
mnist = input_data.read_data_sets('data',one_hot=True)

# 输入节点、输出节点占位
x = tf.placeholder("float",[None,784])
y_ = tf.placeholder("float",[None,10])

# 权重、偏置初始化
def weight_init(shape):
	init_value = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_value)
def bias_init(shape):
	init_value = tf.constant(0.1,shape=shape)
	return tf.Variable(init_value)

W_conv1 = weight_init([5,5,1,32])
b_conv1 = bias_init([32])
W_conv2 = weight_init([5,5,32,64])
b_conv2 = bias_init([64])
W_fc1 = weight_init([7*7*64,1024])
b_fc1 = bias_init([1024])
W_fc2 = weight_init([1024,10])
b_fc2 = bias_init([10])

# 层计算
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2X2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# 损失
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# 参数更新
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 评估
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# session变量初始化
#sess = tf.InteractiveSession() -----  sess.close()
init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)
	
	# 训练
	for i in range(200):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuacy = accuracy.eval(feed_dict={
				x:batch[0],y_:batch[1],keep_prob:1.0})
			print ('step %d, training accuracy %g'%(i,train_accuacy))
		sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

	print ('test accuracy %g'%accuracy.eval(feed_dict={
		x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
