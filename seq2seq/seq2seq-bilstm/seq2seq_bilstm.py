import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np
import matplotlib.pyplot as plt
import helpers

# TASK:      Memorize the provided number sequence and return it again
# TECHNICAL: Use Bidirectional LSTM in the encoder and during training feed generated outputs (greedily)
#            as inputs to decoder

# Constants
PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2  # due to encoder being BiLSTM and decoder being LSTM

# ---- Build model ----
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encode_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

# Encoder
encoder_cell = LSTMCell(encoder_hidden_units)
((encoder_fw_outputs, encoder_bw_outputs),
 (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                                     cell_bw=encoder_cell,
                                                                                     inputs=encoder_inputs_embedded,
                                                                                     time_major=True,
                                                                                     sequence_length=encoder_inputs_length,
                                                                                     dtype=tf.float32)

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

# Decoder
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3  # +2 additional steps, +1 for EOS

eos_time_slice = tf.ones([batch_size], tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)


def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None
    return initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)  # produces tensor shape [batch_size]; says which elements are fin
    finished = tf.reduce_all(elements_finished)  # true if all are finished else false
    next_input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)  # if finished PAD else provide next input
    state = previous_state
    output = previous_output
    loop_state = None
    return elements_finished, next_input, state, output, loop_state


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:  # time = 0
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dims = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dims))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
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
    e_in, e_in_length = helpers.batch(random_lists_raw)
    d_targets, _ = helpers.batch([(sequence) + [EOS] + [PAD] + [PAD] for sequence in random_lists_raw])
    return {
        encoder_inputs: e_in,
        encoder_inputs_length: e_in_length,
        decoder_targets: d_targets,
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
                print(' Sample {0}'.format(i + 1))
                print('  Input     > {0}'.format(inp))
                print('  Predicted > {0}'.format(pred))
                if i > 2:
                    break
            print
