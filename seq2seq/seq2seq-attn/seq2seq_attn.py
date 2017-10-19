import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np
import matplotlib.pyplot as plt
import helpers

# TASK:      Memorize the provided number sequence and return it again @TODO: make this harder (e.g. longer, reverse...)
# TECHNICAL: Use Bidirectional LSTM in the encoder and during training feed generated outputs (greedily)
#            as inputs to decoder, then use attention mechanism (@TODO: specify which one)

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

encoder_outputs = tf.Print(encoder_outputs, [tf.shape(encoder_outputs)], message="Encoder outputs: ")

# Decoder
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3  # +2 additional steps, +1 for EOS

eos_time_slice = tf.ones([batch_size], tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

# TODO: Add attention mechanism, add attention layer, make it work
attention_W = tf.Variable(tf.random_uniform([2 * encoder_hidden_units+decoder_hidden_units, 1]), dtype=tf.float32)
attention_b = tf.Variable(tf.random_uniform([1]), dtype=tf.float32)



def tf_map_multiple(fn, arrays, dtype=tf.float32):
    # applies map in sync over all tensors passed, returns a single tensor
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out


def get_new_state(current_state):

    def get_attn_raw_scalars_over_encoder_hidden(sub_current_state):

        def get_attn_scalars_for_ith_state(i_hidden):
            # Makes a basic 1 layer MLP; current state is the state vectors as a batch, i_hidden is the ith output
            # of the encoder
            totalinput = tf.concat([sub_current_state, i_hidden], axis=1)
            print 'Total input ' + str(totalinput.get_shape()) + ' attention W ' + str(attention_W.get_shape())
            totalinput = tf.Print(totalinput, [tf.shape(i_hidden), tf.shape(totalinput)], message="Ith state, hidden + total")
            return tf.add(tf.matmul(totalinput, attention_W), attention_b)

        # Makes matrix over raw weights for attention alignment
        combined_batch = tf.map_fn(get_attn_scalars_for_ith_state, encoder_outputs)

        print 'Encoder outputs shape ' + str(encoder_outputs.get_shape())
        combined_batch = tf.transpose(combined_batch, perm=[1, 0, 2])
        combined_batch = tf.reshape(combined_batch, shape=[-1, encoder_max_time])
        combined_batch = tf.Print(combined_batch, [tf.shape(combined_batch)], message="Combined batch")
        print 'Combined multiplied batch ' + str(combined_batch.get_shape())
        return combined_batch

    def get_attn_scalars_over_encoder_hidden(current_state_i):
        # Applies softmax over the attention alignment, batchwise
        return tf.nn.softmax(get_attn_raw_scalars_over_encoder_hidden(current_state_i), dim=1)

    # WTF did i do here???
    def get_attn_vectors_from_scalars(current_state_i):
        weights = get_attn_scalars_over_encoder_hidden(current_state_i)
        print 'Attention weights ' + str(weights.get_shape())
        # NOTE: CURRENTLY WE GOT THE DIMS WRONG, AND NEED TO DO SOME PREPROCESSING BEFORE
        encoder_outputs_batch_first = tf.transpose(encoder_outputs, perm=[1, 0, 2])
        encoder_outputs_batch_first = tf.Print(encoder_outputs_batch_first, [tf.shape(encoder_outputs_batch_first)],
                                               message='Encoder batch first shape: ')
        print 'Encoder output batch first shape ' + str(encoder_outputs_batch_first.get_shape())

        def get_weighted_sum_vector(weights, state_vectors):
            # gets a 2d tensor of states and 1d tensor of weights
            def get_weighted_vector(weight, vector):
                print 'Weight ' + str(weight.get_shape()) + ' current vector shape ' + str(vector.get_shape())
                return tf.multiply(weight, vector)

            final_vectors = tf_map_multiple(get_weighted_vector, [weights, state_vectors])
            final_vector = tf.reduce_sum(final_vectors, 1)
            print 'Final vector shape ' + str(final_vector.get_shape())
            final_vector = tf.Print(final_vector, [tf.shape(final_vector)], message="Final vector")
            return final_vector

        weighted_vectors = tf_map_multiple(get_weighted_sum_vector, [weights, encoder_outputs_batch_first])
        print 'Weighted vectors' + str(weighted_vectors.get_shape())
        weighted_vectors = tf.Print(weighted_vectors, [tf.shape(weighted_vectors)], message='Final weighted vector')
        return weighted_vectors

    current_state = tf.Print(current_state, [current_state], message='Current state: ')
    print 'Initial current state ' + str(current_state.get_shape())
    attention_vectors = get_attn_vectors_from_scalars(current_state)
    new_state = tf.concat([current_state, attention_vectors], axis=1)
    new_state = tf.Print(new_state, [tf.shape(new_state)], message='New state: ')
    print 'New state ' + str(new_state.get_shape())
    print ' '
    return new_state


# + 10 for attention vector concatenated at the end
final_W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
final_b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)


def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None#get_new_state(encoder_final_state.c)
    initial_loop_state = None
    return initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    #TODO: look into why previous output is only [batch_size, 40] and not [batch_size, 50]
    #TODO: also look at what type of output is mean, maybe move attention upwards
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, final_W), final_b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)  # produces tensor shape [batch_size]; says which elements are fin
    finished = tf.reduce_all(elements_finished)  # true if all are finished else false
    finished = tf.Print(finished, [finished], message='Finished vector')
    next_input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)  # if finished PAD else provide next input
    state = previous_state
    output = get_new_state(previous_state.c)  # attention concatenated to the end
    output = tf.Print(output, [tf.shape(output)], message='Output shape ')
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
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, final_W), final_b)
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
        sess.run([encoder_outputs], feed)

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
