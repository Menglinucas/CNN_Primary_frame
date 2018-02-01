#########################################################################
# Author: Lin Meng    Time: 2018.2.1	Based on tensorflow frame		#
# Example: dog_cat classification										#
#########################################################################
# This is a primary frame of Convolutional Neural Network with a clear	#
# 	designing process, mainly used for object classification. 			#
# This program can be easily finetuned for the your own purpose.		#													#
# Effective code lines: 67												#
#########################################################################
import dataset
import tensorflow as tf
import numpy as np

# 1. Load the training data
train_path = "E:/pythontest/dog_cat/training_data"	#the path storing training data
img_hsize = 28	#resizing height
img_wsize = 24	#resizing width
classes = ['dogs','cats']	#classes
num_classes = len(classes)	#number of classes
num_channels = 3	#number of channels. Here can be 1 or 3
validation_size = 0.1	#validation data proportion
data_train = dataset.read_train_sets(train_path,img_hsize,img_wsize,num_channels,classes,validation_size)	#load the training data

# 2. Define basic parameters
layers = 5	#total layer number. linkers = layers-1
stride = 1	#filter moving stride
pool = 2	#downsampling times
learning_rate = 1.0e-4	#learning rate of each epoch
epochs = 10000	#total iterations
batch_size = 100	#batch size of training data

# 3. Construct networks
# 3.1  Construct layer: features, variables
# layer1, input layer
features1 = num_channels
x = tf.placeholder("float",[None,img_hsize,img_wsize,features1])
# layer2
features2 = 32
# layer3
features3 = 64
# layer4, fully connected layer
features_fc1 = 800
# layer5, output layer
features_fc2 = num_classes
y_ = tf.placeholder("float",[None,features_fc2])

# 3.2 Construct linkers: filters, weight and bias
def weight_init(shape):	#weight initialization function, wight number ~ nodes
	init_value = tf.truncated_normal(shape,stddev=0.05)
	return tf.Variable(init_value)
def bias_init(shape):	#bias initialization function, bias number ~ latter features
	init_value = tf.constant(0.05,shape=shape)
	return tf.Variable(init_value)
# linker1
filter1 = 5
W_conv1 = weight_init([filter1,filter1,features1,features2])
b_conv1 = bias_init([features2])
# linker2
filter2 = 5
W_conv2 = weight_init([filter2,filter2,features2,features3])
b_conv2 = bias_init([features3])
# linker3
# no filter, flatten process
W_fc1 = weight_init([img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features3,features_fc1])
b_fc1 = bias_init([features_fc1])
#linker4
# no filter
W_fc2 = weight_init([features_fc1,features_fc2])
b_fc2 = bias_init([features_fc2])

# 3.3 Construct operation
def conv2d(x,W,stride):	#convolution function
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
def max_pool_2X2(x,pool):	#pooling function
	return tf.nn.max_pool(x,ksize=[1,pool,pool,1],strides=[1,pool*stride,pool*stride,1],padding='SAME')
# layer1 * W + b = layer2
h_conv1 = tf.nn.relu(conv2d(x,W_conv1,stride)+b_conv1)
h_pool1 = max_pool_2X2(h_conv1,pool)
# layer2 * W + b = layer3
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,stride)+b_conv2)
h_pool2 = max_pool_2X2(h_conv2,pool)
# layer3 * W + b = layer4
h_pool2_flat = tf.reshape(h_pool2,[-1,img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)	#dropout process
# layer4 * W + b = layer5
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# 3.4 Construct loss function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# 3.5 Construct optimizer function used for updating weight and bias
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 3.6 Construct evaluation function used for estimating the accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# 4. Training
# session initialization
init = tf.global_variables_initializer() #sess = tf.InteractiveSession() -----  sess.close()
with tf.Session() as sess:
	sess.run(init)	
	# training
	for i in range(epochs):
		batch = data_train.train.next_batch(batch_size)	#get batch data
		if i%100 == 0:
			train_accuacy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.8})
			print ('step %d, training accuracy %g'%(i,train_accuacy))	#print the accuracy of training set
		sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})	#updating weight and bias
	# print the accuracy of validation set
	print ('test accuracy %g'%accuracy.eval(feed_dict={x: data_train.valid.images, y_: data_train.valid.labels, keep_prob: 1.0}))