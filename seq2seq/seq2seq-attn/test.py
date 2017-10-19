import tensorflow as tf

a = tf.constant([[[5.0], [6.0]], [[7.0], [8.0]]])
a = tf.Print(a, [a.get_shape()], message='Shape of a')

b = tf.reshape(a, [a.get_shape().as_list()[0], a.get_shape().as_list()[1]])
b = tf.Print(b, [b.get_shape()], message='Shape of b')

sess = tf.Session()
sess.run(b)
