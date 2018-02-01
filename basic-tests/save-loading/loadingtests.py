import tensorflow as tf
import numpy
import os
rng = numpy.random


dir = os.path.dirname(os.path.realpath(__file__))


with tf.variable_scope('old_scope'):
    #x1 = tf.get_variable(name='x1', shape=(), initializer=tf.zeros_initializer)
    x1 = tf.contrib.rnn.LSTMCell(num_units=10)
    #saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_scope'))
    saver = tf.train.Saver()

with tf.variable_scope('new_scope'):
    #x2 = tf.get_variable(name='x1', shape=(), initializer=tf.ones_initializer)
    x2 = tf.contrib.rnn.LSTMCell(num_units=10)

for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_scope'):
    print v.name


init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, dir + '/data_1.ckpt')


loader = tf.train.Saver(var_list={'old_scope/x1': x2})

with tf.Session() as sess2:
    loader.restore(sess2, dir + '/data_1.ckpt')
    print sess2.run([x2])
