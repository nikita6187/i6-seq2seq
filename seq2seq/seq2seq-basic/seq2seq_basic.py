import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helpers

# TASK: Memorize the provided number sequence and return it again

# Constants
PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

# ---- Build model ----
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encode_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

# Encoder
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, time_major=True,
                                                         dtype=tf.float32)
del encoder_outputs

# Decoder
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded,
                                                         initial_state=encoder_final_state, dtype=tf.float32,
                                                         time_major=True, scope="plain_decoder")
decoder_logits = tf.contrib.layers.fully_connected(decoder_outputs, vocab_size, activation_fn=None)
decoder_prediction = tf.argmax(decoder_logits, 2)

# Training
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=vocab_size,
                                                                                   dtype=tf.float32),
                                                                 logits=decoder_logits)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()


def next_batch(amount=100):
	random_lists_raw = helpers.generate_random_lists(amount=amount)
	e_in, _ = helpers.batch(random_lists_raw)
	d_targets,_ = helpers.batch([(sequence) + [EOS] for sequence in random_lists_raw])
	d_in,_ = helpers.batch([[EOS] + (sequence) for sequence in random_lists_raw])
	return {
		encoder_inputs: e_in,
		decoder_targets: d_targets,
		decoder_inputs: d_in,
	}


with tf.Session() as sess:
	sess.run(init)
	losses = []
	
	for batch in range(3000):
		feed = next_batch()
		_, l = sess.run([train_op, loss], feed)
		losses.append(l)

		if batch % 100 == 0:
			print('Batch: {0} Loss:{1:2f}'.format(batch, losses[-1]))
			predict = sess.run(decoder_prediction, feed)
			for i, (inp, pred) in enumerate(zip(feed[encoder_inputs].T, predict.T)):
				print(' Sample {0}'.format(i+1))
				print('  Input     > {0}'.format(inp))
				print('  Predicted > {0}'.format(pred))
				if i > 2:
					break
			print
