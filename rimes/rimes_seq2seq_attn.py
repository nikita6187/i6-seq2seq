import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import dataset_loader

# Load batch manager
i, i_l, t, t_l = dataset_loader.load_from_file('train.0010')
bm = dataset_loader.BatchManager(i, i_l, t, t_l, 'EOS', 'PAD')
print bm.lookup

# -- Constants ---
vocab_size = bm.get_size_vocab()
input_dimensions = 20
input_embedding_size = 50
encoder_hidden_units = 256
decoder_hidden_units = encoder_hidden_units * 2  # due to encoder being BiLSTM and decoder being LSTM
attention_hidden_layer_size = 64
max_time = 18

# ---- Build model ----
encoder_inputs = tf.placeholder(shape=(None, None, input_dimensions), dtype=tf.float32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_target_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_target_length')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_full_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_full_length')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = encoder_inputs
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
decoder_inputs_embedded = tf.transpose(decoder_inputs_embedded, perm=[1, 0, 2])  # Make it time major again

# ---- Encoder -------
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
encoder_max_time, batch_size, _ = tf.unstack(tf.shape(encoder_inputs))

eos_time_slice = tf.ones([batch_size], tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.concat([tf.nn.embedding_lookup(embeddings, pad_time_slice), tf.zeros([batch_size, encoder_hidden_units * 2])],
                              axis=1)  # to account for attention

# ----- Attention ------
attention_W_1 = tf.Variable(tf.random_uniform([2 * encoder_hidden_units+decoder_hidden_units, attention_hidden_layer_size]), dtype=tf.float32)
attention_b_1 = tf.Variable(tf.random_uniform([attention_hidden_layer_size]), dtype=tf.float32)
attention_W_2 = tf.Variable(tf.random_uniform([attention_hidden_layer_size, 1], maxval=2), dtype=tf.float32)
attention_b_2 = tf.Variable(tf.random_uniform([1]), dtype=tf.float32)


def tf_map_multiple(fn, arrays, dtype=tf.float32):
    # applies map in sync over all tensors passed, returns a single tensor
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out


def get_attention_from_current_state(current_state):
    #returns attention vectors of shape [batch_size, encoder_hidden_size]

    def get_attn_scalars_over_encoder_hidden(sub_current_state):
        def get_attn_scalars_for_ith_state(i_hidden):
            # Makes a basic 1 layer MLP; current state is the state vectors as a batch, i_hidden is the ith output
            # of the encoder
            # @TODO: try using cos distance
            #totalinput = tf.concat([sub_current_state, i_hidden], axis=1)
            #o1 = tf.nn.relu(tf.add(tf.matmul(totalinput, attention_W_1), attention_b_1))
            #o2 = tf.add(tf.matmul(o1, attention_W_2), attention_b_2)
            o2 = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(sub_current_state, 1),
                                           tf.nn.l2_normalize(i_hidden, 1)), axis=1)
            o2 = tf.Print(o2, [o2], message='o2 can do ', summarize=100)
            return o2

        # Makes matrix over raw weights for attention alignment
        combined_batch = tf.map_fn(get_attn_scalars_for_ith_state, encoder_outputs)  #@TODO: look into shapes + data
        combined_batch = tf.reshape(combined_batch, shape=[-1, encoder_max_time])
        return combined_batch#tf.nn.softmax(combined_batch, dim=1)

    weights = get_attn_scalars_over_encoder_hidden(current_state)
    #weights = tf.Print(weights, [tf.argmax(weights, axis=1)], message='Attention weights ', summarize=10)
    weights = tf.Print(weights, [tf.reduce_max(weights, axis=1)], message='Attention weights 2', summarize=10)
    encoder_outputs_batch_first = tf.transpose(encoder_outputs, perm=[1, 0, 2])

    def get_weighted_sum_vector(weights, state_vectors):
        # gets a 2d tensor of states and 1d tensor of weights
        def get_weighted_vector(weight, vector):
            return tf.multiply(weight, vector)

        final_vectors = tf_map_multiple(get_weighted_vector, [weights, state_vectors])
        final_vector = tf.reduce_sum(final_vectors, 0)
        return final_vector

    weighted_vectors = tf_map_multiple(get_weighted_sum_vector, [weights, encoder_outputs_batch_first])
    return weighted_vectors


final_W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
final_b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

decoder_inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
decoder_inputs_ta = decoder_inputs_ta.unstack(decoder_inputs_embedded)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_target_length)
    attention = get_attention_from_current_state(encoder_final_state.c)
    initial_input = tf.concat([eos_step_embedded, attention], axis=1)
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None
    return initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    attention = get_attention_from_current_state(previous_state.c)
    #output_and_attention = tf.concat([previous_state.h, attention], axis=1)

    def get_next_input():
        #output_logits = tf.add(tf.matmul(previous_output, final_W), final_b)
        #prediction = tf.argmax(output_logits, axis=1)
        #next_input_e = tf.nn.embedding_lookup(embeddings, prediction)
        next_input_e = decoder_inputs_ta.read(time)
        next_input = tf.concat([next_input_e, attention], axis=1)  # add attention to next input
        return next_input

    #elements_finished = (time >= decoder_target_length)  # produces tensor shape [batch_size]; says which elements are fin
    elements_finished = (time >= decoder_full_length)  # produces tensor shape [batch_size]; says which elements are fin
    finished = tf.reduce_all(elements_finished)  # true if all are finished else false
    #@TODO: finish next line
    next_input = tf.cond(tf.reduce_all(time >= decoder_target_length), lambda: pad_step_embedded, get_next_input)  # if finished PAD else provide next input
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

decoder_logits = tf.layers.dense(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(decoder_logits, 2)

# Training
targets_one_hot = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot, logits=decoder_logits)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()

# NOTE: Current hack forces all decoder sequences to go full length
# @TODO: make this more elegant

# @TODO: why are the last 2 weights of attention always 0

def next_batch(batch_manager, amount=4):
    e_in, e_in_length, d_targets, d_targets_length = batch_manager.next_batch(batch_size=amount)
    offset_din, _ = bm.offset(d_targets, bm.lookup_letter(bm.eos))
    d_targets, _ = bm.offset(d_targets, bm.lookup_letter(bm.pad), position=-1)
    return {
        encoder_inputs: np.transpose(e_in, axes=[1, 0, 2]),
        encoder_inputs_length: e_in_length,
        decoder_targets: np.transpose(d_targets),
        decoder_target_length: d_targets_length,
        decoder_inputs: offset_din,
        decoder_full_length: np.asarray([18] * amount),
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
                print '  Decoder input > {0}'.format([bm.get_letter_from_index(x) for x in inp])
                print('  Predicted > {0}'.format(pred))
                print('  Predicted > {0}'.format([bm.get_letter_from_index(x) for x in pred]))
                print('  Target > {0}'.format(target))
                print('  Target > {0}'.format([bm.get_letter_from_index(x) for x in target]))
                if i > 2:
                    break
            print
