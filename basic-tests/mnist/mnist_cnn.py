import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Model building tools
def weight_variable(shape):
	initial = tf.truncated_normal(shape)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.2, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Build model
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))

conv1_w = weight_variable([5, 5, 1, 32])
conv1_b = weight_variable([32])

conv2_w = weight_variable([5, 5, 32, 64])
conv2_b = weight_variable([64])

fc_3_w = weight_variable([7 * 7 * 64, 1024])  # 7, due to max pooling, the initial image will be halved 2 times
fc_3_b = weight_variable([1024])

fc_4_w = weight_variable([1024, 10])
fc_4_b = weight_variable([10])

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = tf.nn.relu(conv2d(x_image, conv1_w) + conv1_b)
conv1_max = max_pool_2x2(conv1)
conv2 = tf.nn.relu(conv2d(conv1_max, conv2_w) + conv2_b)
conv2_max = max_pool_2x2(conv2)
conv2_max_flat = tf.reshape(conv2_max, [-1, 7 * 7 * 64])
fc_3 = tf.nn.relu(tf.matmul(conv2_max_flat, fc_3_w) + fc_3_b)
fc_3_drop = tf.nn.dropout(fc_3, keep_prob)
prediction = tf.matmul(fc_3_drop, fc_4_w) + fc_4_b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

init = tf.global_variables_initializer()

# Measurements
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init)
	
	for i in range(20000):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
		
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
			print('Batch: {0} Train accuracy: {1:3f}'.format(i, train_accuracy))
	
	print('Test accuracy {0:3f}'.format(sess.run(accuracy, feed_dict=
			{x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})))

