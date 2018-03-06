import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core
import dataset_loader

# Load batch manager
i, i_l, t, t_l = dataset_loader.load_from_file('train.0010')
bm = dataset_loader.BatchManager(i, i_l, t, t_l, 'EOS', 'PAD')
print bm.lookup

# -- Constants ---
vocab_size = bm.get_size_vocab()
input_dimensions = 20
input_embedding_size = 50
encoder_hidden_units = 512
decoder_hidden_units = encoder_hidden_units
attention_hidden_layer_size = 64
max_time = 18
batch_size = 2

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


# ---- Encoder ------
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

# Run Dynamic RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    sequence_length=encoder_inputs_length, time_major=True,
    dtype=tf.float32)

decoder_inputs_embedded = tf.Print(decoder_inputs_embedded, [tf.shape(encoder_state)], 'Encoder state shape')

#helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
#            embedding=embeddings,
#            start_tokens=tf.tile([vocab_size-2], [batch_size]),
#            end_token=vocab_size-1)

# ---- Decoder -----
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_inputs_embedded, decoder_full_length, time_major=True)  # TODO: decoder_full_length is bad for training

attention_states = tf.transpose(encoder_outputs, [1, 0, 2])  # attention_states: [batch_size, max_time, num_units]
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    encoder_hidden_units, attention_states,
    memory_sequence_length=encoder_inputs_length)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    tf.contrib.rnn.LSTMCell(decoder_hidden_units),
    attention_mechanism,
    attention_layer_size=decoder_hidden_units)

projection_layer = layers_core.Dense(
    vocab_size, use_bias=False)


decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, decoder_cell.zero_state(tf.size(decoder_full_length), tf.float32),#.clone(cell_state=encoder_state),
    output_layer=projection_layer)

# ---- Training ----
outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
logits = outputs.rnn_output
decoder_prediction = outputs.sample_id


targets_one_hot = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot, logits=logits)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()


# ---- Loader -----


def next_batch(batch_manager, amount=batch_size):
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