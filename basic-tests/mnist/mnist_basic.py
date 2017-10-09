import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Build model
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.matmul(x, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

init = tf.global_variables_initializer()
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Measurements
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init)
	
	for i in range(1000):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={x: batch[0], y: batch[1]})
		
	print('Accuracy {0:3f}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))

