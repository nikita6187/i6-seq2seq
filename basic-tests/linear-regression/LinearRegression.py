import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import random
rng = numpy.random


# Constants
n_samples = 10000
n_epochs = 10


# Generate random data

def function_to_use(x, noise=1):
    return -x + random.uniform(-noise, noise)

def generate_data(amount=10000, min=-10, max=10, scale=1):
    temp_x = []
    temp_y = []
    for i in range(0, amount):
        x = random.uniform(min, max)
        temp_x.append(x)
        temp_y.append(function_to_use(x))
    return temp_x, temp_y


train_x, train_y = generate_data(amount=n_samples)
test_x, test_y = generate_data(amount=100)

# Build model
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(rng.randn(), name="weight")
B = tf.Variable(rng.randn(), name="bias")

prediction = tf.add(tf.multiply(X, W), B)
cost = tf.reduce_sum(tf.pow(prediction-Y, 2)/(2*n_samples))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

var_init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(var_init)

    for epoch in range(n_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if epoch % 10 == 1:
            c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print('Epoch: {0} Cost: {1} W: {2:4f} B: {3:4f}'.format(epoch, c, sess.run(W), sess.run(B)))

    print('Optimization finished!')

    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    print('Training Cost: {0} W: {1:4f} B: {2:4f}'.format(training_cost, sess.run(W), sess.run(B)))

    # Plot
    plot_y = [x * sess.run(W) + sess.run(B) for x in train_x]
    print(plot_y)

    plt.plot(train_x, train_y, 'ro', label='Original Data')
    plt.plot(train_x, plot_y, label='Fitted Line')
    plt.legend()
    plt.show()
