import tensorflow as tf

a = tf.constant(0.1)
b = tf.constant([[3.0, 4.0], [5.0, 6.0]])
c = tf.multiply(a, b)

c = tf.Print(c, [c, a.get_shape()])

sess = tf.Session()
sess.run(c)
