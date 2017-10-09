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

# Build model, tbh its too big
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))

fc_1_w = weight_variable([784, 128])
fc_1_b = bias_variable([128])

fc_2_w = weight_variable([128, 64])
fc_2_b = bias_variable([64])

fc_3_w = weight_variable([64, 10])
fc_3_b = bias_variable([10])

fc_1 = tf.nn.relu(tf.matmul(x, fc_1_w) + fc_1_b)
fc_2 = tf.nn.relu(tf.matmul(fc_1, fc_2_w) + fc_2_b)
prediction = tf.matmul(fc_2, fc_3_w) + fc_3_b

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
		train_step.run(feed_dict={x: batch[0], y: batch[1]})
		
		if i % 100 == 1:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
			print('Batch: {0} Train accuracy: {1:3f}'.format(i, train_accuracy))
			
	print('Test accuracy {0:3f}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
