import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.layers import core as layers_core
import numpy as np
import helpers

# NOTE: Time major

# Constants
input_dimensions = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 64
inputs_embedded = True
transducer_hidden_units = 64
batch_size = 1
GO_SYMBOL = -1
END_SYMBOL = -2

# Helper classes

class LSTMData(object):
    def __init__(self, c, h):
        self.c = c
        self.h = h

class Alignment(object):
    def __init__(self):
        self.alignment_position = (0, 0)                # x = position in target, y = block index
        self.log_prob = 0                               # The sum log prob of this alignment over the target indices
        self.alignment_locations = []                   # At which indices in the target output we need to insert <e>
        self.last_state_transducer = LSTMData(None, None)

    def insert_alignment(self, index, block_index, log_prob):
        self.alignment_locations.append(index)
        self.alignment_position = (index, block_index)
        self.log_prob = log_prob

    def insert_states(self, state_transducer):
        self.last_state_transducer = state_transducer


# Alignment

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
last_encoder_state = LSTMData(None, None)

# TODO: block_index, targets etc. to start at index 1
def run_new_block(session, block_inputs, previous_alignments, block_index, transducer_width, targets, total_blocks,
                  targets_length):
    # TODO: Get encoder outputs and save encoder state                                                      [p]
    # TODO: Run transducer and create transducer_width new alignments for each existing alignments          [p]
        # TODO: Note that the new alignments need to be maximum length of target
    # TODO: Calculate for each alignment the sum log prob
    # TODO: Filter each alignment with the same indices to contain the highest log prob

    def run_encoder(session, encoder_state, inputs):
        """
        Runs the encoder on specified inputs. Returns the outputs and encoder state.
        :param session: Current session.
        :param encoder_state: Current Encoder state.
        :param inputs: The truncated inputs for this block.
        :return: Encoder ouputs [max_time, batch_size, encoder_hidden_units], Encoder state of type LSTMData
        """

        # Inputs
        encoder_inputs = tf.placeholder(shape=(None, None, input_dimensions), dtype=tf.float32, name='encoder_inputs')
        encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        encoder_hidden_c = tf.placeholder(shape=(None, encoder_hidden_units),
                                              dtype=tf.float32, name='encoder_hidden_c')
        encoder_hidden_h = tf.placeholder(shape=(None, encoder_hidden_units),
                                              dtype=tf.float32, name='encoder_hidden_h')

        if inputs_embedded is True:
            encoder_inputs_embedded = encoder_inputs
        else
            encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

        # Build model
        encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
        encoder_hidden_state = LSTMStateTuple(encoder_hidden_c, encoder_hidden_h)
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        encoder_outputs, encoder_hidden_state = tf.nn.dynamic_rnn(
            encoder_cell, encoder_inputs_embedded,
            sequence_length=encoder_inputs_length, time_major=True,
            dtype=tf.float32, initial_state=encoder_hidden_state)

        enc_out, enc_state = session.run([encoder_outputs, encoder_hidden_state],
                                         feed_dict = {
                                             encoder_inputs: inputs,
                                             encoder_inputs_length: inputs.shape[1],
                                             encoder_hidden_c: encoder_state.c,
                                             encoder_hidden_h: encoder_state.h
                                         })

        enc_state = LSTMData(enc_state.c, enc_state.h)  # TODO: see if this is right.
        return enc_out, enc_state

    def run_transducer(session, transducer_state, encoder_outputs, transducer_amount_outputs):
        """
        Runs a transducer on one block of inputs for transducer_amount_outputs.
        :param session: Current session.
        :param transducer_state: The last transducer state as LSTMData object.
        :param encoder_outputs: The outputs of the encoder on the input.
        :param transducer_amount_outputs: The amount of outputs the transducer should produce.
        :return: Transducer logits [transducer_amount_outputs, batch_size(1), vocab_size], Transducer state as LSTMData
        """

        encoder_raw_outputs = tf.placeholder(shape=(None, None, input_dimensions),
                                             dtype=tf.float32,
                                             name='encoder_raw_outputs')
        trans_hidden_c = tf.placeholder(shape=(None, encoder_hidden_units),
                                          dtype=tf.float32, name='trans_hidden_c')
        trans_hidden_h = tf.placeholder(shape=(None, encoder_hidden_units),
                                          dtype=tf.float32, name='trans_hidden_h')

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embeddings,
            start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
            end_token=END_SYMBOL)

        attention_states = tf.transpose(encoder_raw_outputs, [1, 0, 2])  # attention_states: [batch_size, max_time, num_units]

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                encoder_hidden_units, attention_states)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                        tf.contrib.rnn.LSTMCell(transducer_hidden_units),
                        attention_mechanism,
                        attention_layer_size=transducer_hidden_units)

        projection_layer = layers_core.Dense(vocab_size, use_bias=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, LSTMStateTuple(trans_hidden_c, trans_hidden_h),
                    output_layer=projection_layer)

        outputs, transducer_h_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                          output_time_major=True,
                                                          maximum_iterations=transducer_amount_outputs)
        logits = outputs.rnn_output
        decoder_prediction = outputs.sample_id  # For debugging

        trans_out, trans_state = session.run([logits, transducer_h_state],
                                         feed_dict={
                                             encoder_raw_outputs: encoder_outputs,
                                             trans_hidden_c: transducer_state.c,
                                             trans_hidden_h: transducer_state.h
                                         })
        transduce_final_state = LSTMData(trans_state.c, trans_state.h)
        return trans_out, transduce_final_state

    def compute_sum_probabilities(transducer_outputs, targets, alignment, transducer_amount_outputs):
        def get_prob_at_timestep(timestep):
            return np.log(transducer_outputs[timestep][1][targets[start_index + timestep]])

        start_index = alignment.alignment_position[0]  # The current position this alignment is at
        prob = 0
        for i in range(0, transducer_amount_outputs):
            prob += get_prob_at_timestep(i)

        return prob


    # Run encoder and get inputs
    encoder_outputs, encoder_state = run_encoder(session, last_encoder_state, block_inputs)

    # Look into every existing alignment
    for i in range(len(previous_alignments)):
        alignment = previous_alignments[i]

        # Expand the alignment for each transducer width
        min_index = alignment.alignment_position[0] + \
                    max(0, targets_length - ((total_blocks - block_index) * transducer_width) + transducer_width) # TODO: calculate this
        max_index = alignment.alignment_position[0] + transducer_width + min(0, targets_length - (alignment.alignment_position[0] + transducer_width))
        for new_alignment in range(min_index, max_index+1):


# Management

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
