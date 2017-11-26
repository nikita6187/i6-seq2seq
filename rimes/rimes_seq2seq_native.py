import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import numpy as np
import dataset_loader

# Load batch manager
i, t = dataset_loader.load_from_file('train.0010')
bm = dataset_loader.BatchManager(i, t, buckets=[5, 10, 15])
EOS = '-1'
PAD = '-2'
bm.lookup.append(EOS)
bm.lookup.append(PAD)
print bm.lookup

#Constants
tf.set_random_seed(10)
vocab_size = bm.get_size_vocab()
input_embedding_size = 50
encoder_hidden_units = 256
decoder_hidden_units = encoder_hidden_units * 2  # due to encoder being BiLSTM and decoder being LSTM
attention_hidden_layer_size = 128
input_dimensions = 20


encoder_inputs = tf.placeholder(shape=(None, None, input_dimensions), dtype=tf.float32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
#decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')
batch_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')
max_time = tf.placeholder(shape=(), dtype=tf.int32, name='max_time')
decoder_targets_raw = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets_raw')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
decoder_inputs_embedding = tf.nn.embedding_lookup(embeddings, decoder_inputs)

idx = tf.where(tf.not_equal(decoder_targets_raw, 0))
decoder_targets = tf.SparseTensor(idx, tf.gather_nd(decoder_targets_raw, idx), tf.shape(decoder_targets_raw, out_type=tf.int64))



# Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)

# Run Dynamic RNN
#   encoder_outpus: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs,
    sequence_length=encoder_inputs_length, time_major=False,
    dtype=tf.float32)

# Decoder
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedding, decoder_inputs_length, time_major=False)
#helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, bm.lookup_letter(EOS), bm.lookup_letter(PAD))

# Decoding outputs
projection_layer = layers_core.Dense(vocab_size, use_bias=False)
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer, )

decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False)
logits = decoder_outputs.rnn_output
decoder_prediction = decoder_outputs.sample_id

logits = tf.Print(logits, [tf.shape(logits)], message='Logits shape')
logits = tf.Print(logits, [tf.shape(decoder_targets_raw)], message='Decode targ raw shape')

# Loss
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_targets_raw, logits=logits)
#crossent = tf.nn.softmax_cross_entropy_with_logits(labels=decoder_targets, logits=logits)
target_weights = tf.sequence_mask(decoder_inputs_length, max_time, dtype=logits.dtype)
train_loss = (tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size))

# Gradient management & optimization
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2.0)

optimizer = tf.train.AdamOptimizer(0.001)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

init = tf.global_variables_initializer()


def next_batch(batch_manager, amount=4):
    e_in, e_in_length, d_targets, d_targets_length = batch_manager.next_batch(batch_size=amount)
    d_fin_targets = dataset_loader.sparse_tuple_from(d_targets.tolist())

    e_in_length = np.full(amount, e_in.shape[1])
    d_targets_length = np.full(amount, d_targets.shape[1])  # for ctc to work
    offset_din = bm.offset(d_targets, bm.lookup_letter(EOS))

    return {
        encoder_inputs: e_in,#np.transpose(e_in, axes=[1, 0, 2]),
        encoder_inputs_length: e_in_length,
        decoder_inputs_length: d_targets_length,
        #decoder_targets : d_targets,
        decoder_targets_raw: d_targets,
        batch_size: amount,
        max_time: d_targets.shape[1],
        decoder_inputs: offset_din,
    }

with tf.Session() as sess:
    sess.run(init)
    losses = []

    for batch in range(20000):
        feed = next_batch(batch_manager=bm)
        _, l, predict = sess.run([update_step, train_loss, decoder_prediction], feed)
        losses.append(l)

        if batch % 1 == 0:
            print('Batch: {0} Loss:{1:2f}'.format(batch, losses[-1]))
            for i, (inp, pred, target) in enumerate(zip(feed[encoder_inputs].T, predict, feed[decoder_targets_raw])):
                print(' Sample {0}'.format(i + 1))
                # print('  Input     > {0}'.format(inp))
                print('  Predicted > {0}'.format(pred))
                print('  Predicted > {0}'.format([bm.get_letter_from_index(x) for x in pred]))
                print('  Target > {0}'.format(target))
                print('  Target > {0}'.format([bm.get_letter_from_index(x) for x in target]))
                if i > 2:
                    break
            print


