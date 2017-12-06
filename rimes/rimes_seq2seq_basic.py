import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import dataset_loader

# Load batch manager
i, i_l, t, t_l = dataset_loader.load_from_file('train.0010')
bm = dataset_loader.BatchManager(i, i_l, t, t_l, 'EOS', 'PAD')
print bm.lookup


# Constants
vocab_size = bm.get_size_vocab() + 1
encoder_hidden_units = 512
decoder_hidden_units = encoder_hidden_units
input_dimensions = 20
input_embedding_size = input_dimensions

# ---- Build model ----
encoder_inputs = tf.placeholder(shape=(None, None, input_dimensions), dtype=tf.float32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

# Encoder
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, time_major=True, dtype=tf.float32)
del encoder_outputs

# Decoder
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_inputs_embedded = tf.transpose(decoder_inputs_embedded, perm=[1, 0, 2])
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


def next_batch(batch_manager, amount=32):
    e_in, e_in_length, d_targets, d_targets_length = batch_manager.next_batch(batch_size=amount)
    offset_din, _ = bm.offset(d_targets, bm.lookup_letter(bm.eos))
    d_targets, _ = bm.offset(d_targets, bm.lookup_letter(bm.pad), position=-1)
    return {
        encoder_inputs: np.transpose(e_in, axes=[1, 0, 2]),
        decoder_targets: np.transpose(d_targets),
        decoder_inputs: offset_din,
    }

with tf.Session() as sess:
    sess.run(init)
    losses = []

    for batch in range(20000):
        feed = next_batch(batch_manager=bm)
        _, l, predict = sess.run([train_op, loss, decoder_prediction], feed)
        losses.append(l)

        if batch % 1 == 0:
            print('Batch: {0} Loss:{1:2f}'.format(batch, losses[-1]))
            for i, (inp, pred, target) in enumerate(zip(feed[decoder_inputs], predict.T, feed[decoder_targets].T)):
                print(' Sample {0}'.format(i + 1))
                print('  Decoder input > {0}').format([bm.get_letter_from_index(x) for x in inp])
                print('  Predicted > {0}'.format(pred))
                print('  Predicted > {0}'.format([bm.get_letter_from_index(x) for x in pred]))
                print('  Target > {0}'.format(target))
                print('  Target > {0}'.format([bm.get_letter_from_index(x) for x in target]))
                if i > 2:
                    break
            print

