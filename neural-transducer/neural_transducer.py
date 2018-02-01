import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.layers import core as layers_core
import numpy as np
import copy
import os

# Implementation of the "A Neural Transducer" paper, Navdeep Jaitly et. al (2015): https://arxiv.org/abs/1511.04868

# NOTE: Time major

# Constants
input_dimensions = 1
vocab_size = 5
input_embedding_size = 20
encoder_hidden_units = 8
inputs_embedded = True
transducer_hidden_units = 8
batch_size = 1  # Cannot be increased, see paper
GO_SYMBOL = vocab_size - 1  # TODO: Make these constants correct
E_SYMBOL = vocab_size - 2
input_block_size = 3
log_prob_init_value = 0
beam_width = 5  # For inference
dir = os.path.dirname(os.path.realpath(__file__))


# ---------------- Helper classes -----------------------

class Alignment(object):
    def __init__(self):
        self.alignment_position = (0, 1)  # x = position in target (y~), y = block index, both start at 1
        self.log_prob = log_prob_init_value  # The sum log prob of this alignment over the target indices
        self.alignment_locations = []  # At which indices in the target output we need to insert <e>
        self.last_state_transducer = np.zeros(shape=(2, 1, transducer_hidden_units))  # Transducer state

    def __compute_sum_probabilities(self, transducer_outputs, targets, transducer_amount_outputs):
        def get_prob_at_timestep(timestep):
            return np.log(transducer_outputs[timestep][0][targets[start_index + timestep]])

        start_index = self.alignment_position[0] - transducer_amount_outputs  # The current position of this alignment
        prob = log_prob_init_value
        for i in range(0, transducer_amount_outputs):  # Do not include e symbol in calculation
            prob += get_prob_at_timestep(i)
        return prob

    def insert_alignment(self, index, block_index, transducer_outputs, targets, transducer_amount_outputs,
                         new_transducer_state):
        """
        Inserts alignment properties for a new block.
        :param index: The index of of y~ corresponding to the last target index.
        :param block_index: The new block index.
        :param transducer_outputs: The computed transducer outputs.
        :param targets: The complete target array, should be of shape [total_target_length].
        :param transducer_amount_outputs: The amount of outputs that the transducer created in this block.
        :param new_transducer_state: The new transducer state of shape [2, 1, transducer_hidden_units]
        :return:
        """
        self.alignment_locations.append(index)
        self.alignment_position = (index, block_index)
        # TODO: look if new log_prob is done additively or absolute (I think additively)
        self.log_prob += self.__compute_sum_probabilities(transducer_outputs, targets, transducer_amount_outputs)
        self.last_state_transducer = new_transducer_state


# ----------------- Model -------------------------------


class Model(object):
    def __init__(self):
        self.var_list = []
        self.max_blocks, self.inputs_full_raw, self.transducer_list_outputs, self.start_block, self.encoder_hidden_init,\
            self.trans_hidden_init, self.logits, self.encoder_hidden_state_new, \
            self.transducer_hidden_state_new, self.train_saver = self.build_full_transducer()

        self.targets, self.train_op, self.loss = self.build_training_step()

    def build_full_transducer(self):
        with tf.variable_scope('transducer_training'):

            embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0),
                                     dtype=tf.float32,
                                     name='embedding')
            # Inputs
            max_blocks = tf.placeholder(dtype=tf.int32, name='max_blocks')  # total amount of blocks to go through
            inputs_full_raw = tf.placeholder(shape=(None, batch_size, input_dimensions), dtype=tf.float32,
                                             name='inputs_full_raw')  # shape [max_time, 1, input_dims]
            transducer_list_outputs = tf.placeholder(shape=(None,), dtype=tf.int32,
                                                     name='transducer_list_outputs')  # amount to output per block
            start_block = tf.placeholder(dtype=tf.int32, name='transducer_start_block')  # where to start the input

            encoder_hidden_init = tf.placeholder(shape=(2, 1, encoder_hidden_units), dtype=tf.float32,
                                                 name='encoder_hidden_init')
            trans_hidden_init = tf.placeholder(shape=(2, 1, transducer_hidden_units), dtype=tf.float32,
                                               name='trans_hidden_init')

            # Temporary constants, maybe changed during inference
            end_symbol = tf.get_variable(name='end_symbol', initializer=tf.constant_initializer(vocab_size),
                                         shape=(), dtype=tf.int32)

            # Turn inputs into tensor which is easily readable
            inputs_full = tf.reshape(inputs_full_raw, shape=[-1, input_block_size, batch_size, input_dimensions])

            # Outputs
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_blocks)

            init_state = (start_block, outputs_ta, encoder_hidden_init, trans_hidden_init)

            # Initiate cells, NOTE: if there is a future error, put these back inside the body function
            encoder_cell = tf.contrib.rnn.LSTMCell(num_units=encoder_hidden_units)
            transducer_cell = tf.contrib.rnn.LSTMCell(transducer_hidden_units)

            def cond(current_block, outputs_int, encoder_hidden, trans_hidden):
                return current_block < start_block + max_blocks

            def body(current_block, outputs_int, encoder_hidden, trans_hidden):

                # --------------------- ENCODER ----------------------------------------------------------------------
                encoder_inputs = inputs_full[current_block - start_block]
                encoder_inputs_length = [tf.shape(encoder_inputs)[0]]
                encoder_hidden_state = encoder_hidden

                if inputs_embedded is True:
                    encoder_inputs_embedded = encoder_inputs
                else:
                    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

                # Build model

                # Build previous state
                encoder_hidden_c, encoder_hidden_h = tf.split(encoder_hidden_state, num_or_size_splits=2, axis=0)
                encoder_hidden_c = tf.reshape(encoder_hidden_c, shape=[-1, encoder_hidden_units])
                encoder_hidden_h = tf.reshape(encoder_hidden_h, shape=[-1, encoder_hidden_units])
                encoder_hidden_state_t = LSTMStateTuple(encoder_hidden_c, encoder_hidden_h)

                #   encoder_outputs: [max_time, batch_size, num_units]
                encoder_outputs, encoder_hidden_state_new = tf.nn.dynamic_rnn(
                    encoder_cell, encoder_inputs_embedded,
                    sequence_length=encoder_inputs_length, time_major=True,
                    dtype=tf.float32, initial_state=encoder_hidden_state_t)

                # Modify output of encoder_hidden_state_new so that it can be fed back in again without problems.
                encoder_hidden_state_new = tf.concat([encoder_hidden_state_new.c, encoder_hidden_state_new.h], axis=0)
                encoder_hidden_state_new = tf.reshape(encoder_hidden_state_new, shape=[2, -1, encoder_hidden_units])

                # --------------------- TRANSDUCER --------------------------------------------------------------------
                encoder_raw_outputs = encoder_outputs
                trans_hidden_state = trans_hidden  # Save/load the state as one tensor
                transducer_amount_outputs = transducer_list_outputs[current_block - start_block]

                # Model building
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embeddings,
                    start_tokens=tf.tile([GO_SYMBOL], [batch_size]),  # TODO: check if this looks good
                    end_token=end_symbol)  # vocab size, so that it doesn't prematurely end the decoding

                attention_states = tf.transpose(encoder_raw_outputs,
                                                [1, 0, 2])  # attention_states: [batch_size, max_time, num_units]

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    encoder_hidden_units, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    transducer_cell,
                    attention_mechanism,
                    attention_layer_size=transducer_hidden_units)

                projection_layer = layers_core.Dense(vocab_size, use_bias=False)

                # Build previous state
                trans_hidden_c, trans_hidden_h = tf.split(trans_hidden_state, num_or_size_splits=2, axis=0)
                trans_hidden_c = tf.reshape(trans_hidden_c, shape=[-1, transducer_hidden_units])
                trans_hidden_h = tf.reshape(trans_hidden_h, shape=[-1, transducer_hidden_units])
                trans_hidden_state_t = LSTMStateTuple(trans_hidden_c, trans_hidden_h)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper,
                    decoder_cell.zero_state(1, tf.float32).clone(cell_state=trans_hidden_state_t),
                    output_layer=projection_layer)

                outputs, transducer_hidden_state_new, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                            output_time_major=True,
                                                                                            maximum_iterations=transducer_amount_outputs)
                logits = outputs.rnn_output  # logits of shape [max_time,batch_size,vocab_size]
                decoder_prediction = outputs.sample_id  # For debugging

                # Modify output of transducer_hidden_state_new so that it can be fed back in again without problems.
                transducer_hidden_state_new = tf.concat(
                    [transducer_hidden_state_new[0].c, transducer_hidden_state_new[0].h],
                    axis=0)
                transducer_hidden_state_new = tf.reshape(transducer_hidden_state_new,
                                                         shape=[2, -1, transducer_hidden_units])

                # Note the outputs
                outputs_int = outputs_int.write(current_block - start_block, logits)

                return current_block + 1, outputs_int, encoder_hidden_state_new, transducer_hidden_state_new

            _, outputs_final, encoder_hidden_state_new, transducer_hidden_state_new = \
                tf.while_loop(cond, body, init_state, parallel_iterations=1)

            # Process outputs
            outputs = outputs_final.concat()
            logits = tf.reshape(outputs, shape=(-1, 1, vocab_size))  # And now its [max_output_time, batch_size, vocab]

            # For loading the model later on
            logits = tf.identity(logits, name='logits')
            encoder_hidden_state_new = tf.identity(encoder_hidden_state_new, name='encoder_hidden_state_new')
            transducer_hidden_state_new = tf.identity(encoder_hidden_state_new, name='transducer_hidden_state_new')

        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transducer_training'):
            print v.name
        print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transducer_training')

        #self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transducer_training')
        #train_saver = tf.train.Saver(var_list=self.var_list)
        train_saver = tf.train.Saver()  # For now save everything

        return max_blocks, inputs_full_raw, transducer_list_outputs, start_block, encoder_hidden_init,\
            trans_hidden_init, logits, encoder_hidden_state_new, transducer_hidden_state_new, train_saver

    def build_training_step(self):
        targets = tf.placeholder(shape=(None,), dtype=tf.int32, name='targets')
        targets_one_hot = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)

        targets_one_hot = tf.Print(targets_one_hot, [targets_one_hot], message='Targets: ', summarize=100)
        targets_one_hot = tf.Print(targets_one_hot, [self.logits], message='Logits: ', summarize=100)

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot,
                                                                         logits=self.logits)
        loss = tf.reduce_mean(stepwise_cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)
        return targets, train_op, loss

    def __create_loading_dic(self, list_vars, old_name, new_name):
        dic = {}
        for var in list_vars:
            dic[var.name] = var.name.replace(old_name, new_name)
        return dic

    def build_beamsearch_inference_transducer(self):
        with tf.variable_scope('transducer_inference'):
            embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0),
                                     dtype=tf.float32,
                                     name='embedding')

            # shape [input_block_size, 1, input_dims], inputs for this block!
            inputs_full_raw = tf.placeholder(shape=(input_block_size, batch_size, input_dimensions), dtype=tf.float32,
                                             name='inputs_full_raw')

            trans_max_outputs = tf.placeholder(shape=(), dtype=tf.int32, name='transducer_list_outputs')

            encoder_hidden_init = tf.placeholder(shape=(2, 1, encoder_hidden_units), dtype=tf.float32,
                                                 name='encoder_hidden_init')
            trans_hidden_init = tf.placeholder(shape=(2, 1, transducer_hidden_units), dtype=tf.float32,
                                               name='trans_hidden_init')

            # Initiate cells
            encoder_cell = tf.contrib.rnn.LSTMCell(num_units=encoder_hidden_units)
            transducer_cell = tf.contrib.rnn.LSTMCell(num_units=transducer_hidden_units)

            # --------------------- ENCODER ----------------------------------------------------------------------
            encoder_inputs = inputs_full_raw
            encoder_inputs_length = [tf.shape(encoder_inputs)[0]]
            encoder_hidden_state = encoder_hidden_init

            if inputs_embedded is True:
                encoder_inputs_embedded = encoder_inputs
            else:
                encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

            # Build previous state
            encoder_hidden_c, encoder_hidden_h = tf.split(encoder_hidden_state, num_or_size_splits=2, axis=0)
            encoder_hidden_c = tf.reshape(encoder_hidden_c, shape=[-1, encoder_hidden_units])
            encoder_hidden_h = tf.reshape(encoder_hidden_h, shape=[-1, encoder_hidden_units])
            encoder_hidden_state_t = LSTMStateTuple(encoder_hidden_c, encoder_hidden_h)

            #   encoder_outputs: [max_time, batch_size, num_units]
            encoder_outputs, encoder_hidden_state_new = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs_embedded,
                sequence_length=encoder_inputs_length, time_major=True,
                dtype=tf.float32, initial_state=encoder_hidden_state_t)

            # Modify output of encoder_hidden_state_new so that it can be fed back in again without problems.
            encoder_hidden_state_new = tf.concat([encoder_hidden_state_new.c, encoder_hidden_state_new.h], axis=0)
            encoder_hidden_state_new = tf.reshape(encoder_hidden_state_new, shape=[2, -1, encoder_hidden_units])

            # --------------------- TRANSDUCER --------------------------------------------------------------------
            encoder_raw_outputs = encoder_outputs
            trans_hidden_state = trans_hidden_init  # Save/load the state as one tensor
            transducer_amount_outputs = trans_max_outputs

            # Model building
            attention_states = tf.transpose(encoder_raw_outputs,
                                            [1, 0, 2])  # attention_states: [batch_size, max_time, num_units]
            attention_states = tf.contrib.seq2seq.tile_batch(attention_states, multiplier=beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                encoder_hidden_units, attention_states)

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                transducer_cell,
                attention_mechanism,
                attention_layer_size=transducer_hidden_units)

            projection_layer = layers_core.Dense(vocab_size, use_bias=False)

            # Build previous state
            trans_hidden_c, trans_hidden_h = tf.split(trans_hidden_state, num_or_size_splits=2, axis=0)
            trans_hidden_c = tf.reshape(trans_hidden_c, shape=[-1, transducer_hidden_units])
            trans_hidden_h = tf.reshape(trans_hidden_h, shape=[-1, transducer_hidden_units])
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(
                tf.contrib.seq2seq.tile_batch(trans_hidden_c, multiplier=beam_width),
                tf.contrib.seq2seq.tile_batch(trans_hidden_h, multiplier=beam_width))

            decoder_init_state_inf = decoder_cell.zero_state(dtype=tf.float32, batch_size=1 * beam_width). \
                clone(cell_state=initial_state)

            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, embeddings,
                                                                     start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
                                                                     end_token=E_SYMBOL,
                                                                     initial_state=decoder_init_state_inf,
                                                                     beam_width=beam_width,
                                                                     output_layer=projection_layer)

            outputs, transducer_hidden_state_new, seq_len = tf.contrib.seq2seq.dynamic_decode(
                inference_decoder,
                output_time_major=True,
                maximum_iterations=transducer_amount_outputs)

            logits = outputs.beam_search_decoder_output.scores  # score of shape all beams
            decoder_prediction = outputs.predicted_ids  # For debugging

        # TODO: solve loading
        # TODO: create dictionary that maps transducer_training scope to transducer_inference scope
        # dic = self.__create_loading_dic(self.var_list, 'transducer_training', 'transducer_inference')
        dic = {
            'transducer_training/embedding': embeddings,
            'transducer_training/rnn/lstm_cell': encoder_cell,
            'transducer_training/memory_layer': attention_mechanism,
            'transducer_training/decoder/attention_wrapper': decoder_cell,
            'transducer_training/decoder/dense': projection_layer
        }

        inference_loader = tf.train.Saver(var_list=dic)
        return inputs_full_raw, trans_max_outputs, encoder_hidden_init, trans_hidden_init, encoder_hidden_state_new, \
            transducer_hidden_state_new, logits, decoder_prediction, inference_loader


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


model = Model()


# ----------------- Alignment --------------------------

def get_alignment(session, inputs, targets, input_block_size, transducer_max_width):
    """
    Finds the alignment of the target sequence to the actual output.
    :param session: The current session.
    :param inputs: The complete inputs for the encoder of shape [max_time, 1, input_dimensions], note padding if needed
    :param targets: The target sequence of shape [time] where each enty is an index.
    :param input_block_size: The width of one encoder block.
    :param transducer_max_width: The max width of one transducer block.
    :return: Returns a list of indices where <e>'s need to be inserted into the target sequence. (see paper)
    """

    def run_new_block(session, full_inputs, previous_alignments, block_index, transducer_max_width, targets,
                      total_blocks, last_encoder_state):
        """
        Runs one block of the alignment process.
        :param session: The current TF session.
        :param full_inputs: The full inputs. Shape: [max_time, 1, input_dimensions]
        :param previous_alignments: List of alignment objects from previous block step.
        :param block_index: The index of the current new block.
        :param transducer_max_width: The max width of the transducer block.
        :param targets: The full target array of shape [time]
        :param total_blocks: The total amount of blocks.
        :param last_encoder_state: The encoder state of the previous step. Shape [2, 1, encoder_hidden_units]
        :return: new_alignments as list of Alignment objects,
        last_encoder_state_new in shape of [2, 1, encoder_hidden_units]
        """

        last_encoder_state_new = last_encoder_state  # fallback value

        def run_transducer(session, inputs_full, encoder_state, transducer_state, transducer_width):
            """
            Runs a transducer on one block of inputs for transducer_amount_outputs.
            :param session: Current session.
            :param inputs_full: The full inputs. Shape: [max_time, 1, input_dimensions]
            :param transducer_state: The last transducer state as [2, 1, transducer_hidden_units] tensor.
            :param transducer_width: The amount of outputs the transducer should produce.
            :return: transducer outputs [max_output_time, 1, vocab], transducer_state [2, 1, transducer_hidden_units],
            encoder_state [2, 1, encoder_hidden_units]
            """
            logits, trans_state, enc_state = session.run([model.logits, model.transducer_hidden_state_new,
                                                             model.encoder_hidden_state_new],
                                                 feed_dict={
                                                     model.inputs_full_raw: inputs_full,
                                                     model.max_blocks: 1,
                                                     model.transducer_list_outputs: [transducer_width],
                                                     model.start_block: block_index,
                                                     model.encoder_hidden_init: encoder_state,
                                                     model.trans_hidden_init: transducer_state,
                                                 })
            # apply softmax on the outputs
            trans_out = softmax(logits, axis=2)

            return trans_out, trans_state, enc_state

        # Look into every existing alignment
        new_alignments = []
        for i in range(len(previous_alignments)):
            alignment = previous_alignments[i]

            # Expand the alignment for each transducer width, only look at valid options
            targets_length = len(targets)
            min_index = alignment.alignment_position[0] + transducer_max_width + \
                        max(-transducer_max_width,
                            targets_length - ((total_blocks - block_index + 1) * transducer_max_width
                                              + alignment.alignment_position[0]))
            max_index = alignment.alignment_position[0] + transducer_max_width + min(0, targets_length - (
                    alignment.alignment_position[0] + transducer_max_width))

            # new_alignment_index's value is equal to the index of y~ for that computation
            for new_alignment_index in range(min_index, max_index + 1):  # +1 so that the max_index is also used
                # print 'Alignment index: ' + str(new_alignment_index)
                # Create new alignment
                new_alignment = copy.deepcopy(alignment)
                new_alignment_width = new_alignment_index - new_alignment.alignment_position[0]
                trans_out, trans_state, last_encoder_state_new = run_transducer(session=session,
                                                                                inputs_full=full_inputs,
                                                                                encoder_state=last_encoder_state,
                                                                                transducer_state=alignment.last_state_transducer,
                                                                                transducer_width=new_alignment_width+1)
                # last_encoder_state_new being set every time again -> not relevant

                new_alignment.insert_alignment(new_alignment_index, block_index, trans_out, targets,
                                               new_alignment_width, trans_state)
                new_alignments.append(new_alignment)

        # Delete all overlapping alignments, keeping the highest log prob
        for a in reversed(new_alignments):
            for o in new_alignments:
                if o is not a and a.alignment_position == o.alignment_position and o.log_prob > a.log_prob:
                    if a in new_alignments:
                        new_alignments.remove(a)

        return new_alignments, last_encoder_state_new

    # Manage variables
    amount_of_input_blocks = int(np.ceil(inputs.shape[0] / input_block_size))
    current_block_index = 1
    current_alignments = [Alignment()]
    last_encoder_state = np.zeros(shape=(2, 1, encoder_hidden_units))

    # Do assertions to check whether everything was correctly set up.
    assert inputs.shape[0] % input_block_size == 0, \
        'Input shape not corresponding to input block size (add padding or see if batch first).'
    assert inputs.shape[2] == input_dimensions, 'Input dimension [2] not corresponding to specified input dimension.'
    assert inputs.shape[1] == 1, 'Batch size needs to be one.'
    assert transducer_max_width * amount_of_input_blocks >= len(targets), 'transducer_max_width to small for targets'

    for block in range(current_block_index, amount_of_input_blocks + 1):
        # Run all blocks
        current_alignments, last_encoder_state = run_new_block(session=session, full_inputs=inputs,
                                                               previous_alignments=current_alignments,
                                                               block_index=block,
                                                               transducer_max_width=transducer_max_width,
                                                               targets=targets, total_blocks=amount_of_input_blocks,
                                                               last_encoder_state=last_encoder_state)

    # Check if we've found an alignment, it should be one
    assert len(current_alignments) == 1

    return current_alignments[0].alignment_locations

# ----------------- Training --------------------------


def apply_training_step(session, inputs, targets, input_block_size, transducer_max_width):
    """
    Applies a training step to the transducer model. This method can be called multiple times from e.g. a loop.
    :param session: The current session.
    :param inputs: The full inputs. Shape: [max_time, 1, input_dimensions]
    :param targets: The full targets. Shape: [time]. Each entry is an index.
    :param input_block_size: The block width for the inputs.
    :param transducer_max_width: The max width for the transducer. Not including the output symbol <e>
    :return: Loss of this training step.
    """

    # Get alignment and insert it into the targets
    alignment = get_alignment(session=session, inputs=inputs, targets=targets, input_block_size=input_block_size,
                              transducer_max_width=transducer_max_width)
    print alignment

    offset = 0
    for e in alignment:
        targets.insert(e+offset, E_SYMBOL)
        offset += 1

    # Calc length for each transducer block
    lengths = []
    alignment.insert(0, 0)  # This is so that the length calculation is done correctly
    for i in range(1, len(alignment)):
        lengths.append(alignment[i] - alignment[i-1] + 1)

    print lengths

    # Init values
    encoder_hidden_init = np.zeros(shape=(2, 1, encoder_hidden_units))
    trans_hidden_init = np.zeros(shape=(2, 1, transducer_hidden_units))

    # Run training step
    _, loss = sess.run([model.train_op, model.loss], feed_dict={
        model.max_blocks: len(lengths),
        model.inputs_full_raw: inputs,
        model.transducer_list_outputs: lengths,
        model.targets: targets,
        model.start_block: 0,
        model.encoder_hidden_init: encoder_hidden_init,
        model.trans_hidden_init: trans_hidden_init
    })

    return loss


def save_model_for_inference(session, path_name):
    model.train_saver.save(session, path_name)
    print 'Model saved to ' + str(path_name)


# ----------------- Inference --------------------------

# TODO: load in previously built model
# TODO: change model to allow optional usage of beam search decoder [p]
# TODO: add beam search with score based on log softmax addition
# TODO: select best one at the end
# TODO: documentation

class InferenceManager(object):

    # session is only needed for greedy inference, path points to model without any special ending
    def __init__(self, beam_search, path, session, transducer_width):
        self.transducer_width = transducer_width
        self.beam_search = beam_search
        if beam_search is True:
            self.inputs_full_raw, self.trans_max_outputs, self.encoder_hidden_init, self.trans_hidden_init, \
                self.encoder_hidden_state_new, self.transducer_hidden_state_new, self.logits, \
                self.step_prediction, self.inference_beam_loader = self.build_beam_inference()
        else:
            self.max_blocks, self.inputs_full_raw, self.transducer_list_outputs, self.start_block, self.encoder_hidden_init, \
                self.trans_hidden_init, self.logits, self.encoder_hidden_state_new, \
                self.transducer_hidden_state_new = self.build_greedy_inference(path=path, session=session)

    def build_beam_inference(self):
        inputs_full_raw, trans_max_outputs, encoder_hidden_init, trans_hidden_init, encoder_hidden_state_new, \
            transducer_hidden_state_new, logits, prediction, inference_loader = model.build_beamsearch_inference_transducer()
        return inputs_full_raw, trans_max_outputs, encoder_hidden_init, trans_hidden_init, encoder_hidden_state_new, \
            transducer_hidden_state_new, logits, prediction, inference_loader

    def build_greedy_inference(self, path, session):
        # Restore graph
        saver = tf.train.import_meta_graph(path + '.meta')
        saver.restore(session, path)
        # Setup constants
        graph = tf.get_default_graph()
        end_symbol = graph.get_tensor_by_name(name='transducer_training/end_symbol:0')
        end_symbol = tf.assign(end_symbol, value=E_SYMBOL)
        session.run(end_symbol)
        # Get inputs
        max_blocks = graph.get_tensor_by_name(name='transducer_training/max_blocks:0')
        inputs_full_raw = graph.get_tensor_by_name(name='transducer_training/inputs_full_raw:0')
        transducer_list_outputs = graph.get_tensor_by_name(name='transducer_training/transducer_list_outputs:0')
        start_block = graph.get_tensor_by_name(name='transducer_training/transducer_start_block:0')
        encoder_hidden_init = graph.get_tensor_by_name(name='transducer_training/encoder_hidden_init:0')
        trans_hidden_init = graph.get_tensor_by_name(name='transducer_training/trans_hidden_init:0')
        # Get return ops
        logits = graph.get_operation_by_name(name='transducer_training/logits').outputs[0]
        encoder_hidden_state_new = graph.get_operation_by_name(name='transducer_training/encoder_hidden_state_new').outputs[0]
        transducer_hidden_state_new = graph.get_operation_by_name(name='transducer_training/transducer_hidden_state_new').outputs[0]

        return max_blocks, inputs_full_raw, transducer_list_outputs, start_block, encoder_hidden_init, \
            trans_hidden_init, logits, encoder_hidden_state_new, transducer_hidden_state_new

    def run_inference(self, session, model_path, full_inputs):

        def run_greedy_transducer_block(session, full_inputs, current_block, encoder_init_state, transducer_init_state):
            logits, encoder_new_state, transducer_new_state = \
                session.run([self.logits, self.encoder_hidden_state_new,
                             self.transducer_hidden_state_new], feed_dict={
                    self.inputs_full_raw: full_inputs,
                    self.max_blocks: 1,
                    self.transducer_list_outputs: [self.transducer_width],
                    self.start_block: current_block,
                    self.encoder_hidden_init: encoder_init_state,
                    self.trans_hidden_init: transducer_init_state
                })

            print 'New info:'
            print logits
            print encoder_new_state
            print transducer_new_state

            return logits, encoder_new_state, transducer_new_state

        if self.beam_search is True:
            self.inference_beam_loader.restore(sess=session, save_path=model_path)

        # Meta parameters
        amount_of_input_blocks = int(np.ceil(full_inputs.shape[0] / input_block_size))
        # Init encoder/decoder states
        last_encoder_state = np.zeros(shape=(2, batch_size, encoder_hidden_units))
        last_transducer_state = np.zeros(shape=(2, batch_size, transducer_hidden_units))
        logits = []

        for current_input_block in range(0, amount_of_input_blocks):
            if self.beam_search is False:
                new_logits, last_encoder_state, last_transducer_state = \
                    run_greedy_transducer_block(session=session,
                                                full_inputs=full_inputs,
                                                current_block=current_input_block,
                                                encoder_init_state=last_encoder_state,
                                                transducer_init_state=last_transducer_state)
                logits.append(new_logits)

        # Post process logits into one np array
        logit_arr = np.concatenate(logits, axis=0)  # Is now of shape [max_time, batch_size, vocab_size] (THEORITCALLY)

        return logit_arr

# ---------------------- Testing -----------------------------

testy_targets = np.asarray([1, 1, 1, 1, 1])


def test_new_block():
    def run_block(block_index, prev):
        na, _ = run_new_block(None, block_inputs=None, previous_alignments=prev, block_index=block_index,
                              transducer_max_width=3, targets=testy_targets, total_blocks=4, last_encoder_state=None)
        return na

    na = [Alignment()]
    for i in range(0, 3):
        na = run_block(i + 2, na)
        print 'Round ' + str(i + 1) + ' -----------'
        for a in na:
            print 'Alignment: ' + str(a.alignment_position)
            print a.log_prob
            print a.alignment_locations
        print ''


# Testing the alignment class
def test_alignment_class():
    testyAlignment = Alignment()
    testy_outputs = np.asarray([[[0.1, 0.7, 0.2]], [[0.2, 0.1, 0.7]]])
    testyAlignment.insert_alignment(2, 1, testy_outputs, testy_targets, 2, None)
    print 'Log prob for test 1: ' + str(testyAlignment.log_prob)  # Should be: -2.65926003693
    testy_outputs = np.asarray([[[0.2, 0.7, 0.1]], [[0.3, 0.1, 0.6]], [[0.2, 0.1, 0.7]]])
    testyAlignment.insert_alignment(5, 2, testy_outputs, testy_targets, 3, None)
    print 'Log prob for test 2: ' + str(testyAlignment.log_prob)  # Should be: -4.96184512993
    print testyAlignment.alignment_locations


def test_get_alignment(sess):
    testy_inputs = np.random.uniform(-1.0, 1.0, size=(12, 1, input_dimensions))
    print get_alignment(sess, inputs=testy_inputs, targets=testy_targets, input_block_size=input_block_size,
                        transducer_max_width=2)


# ---------------------- Management -----------------------------

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # test_get_alignment(sess)

    # Apply training step
    for i in range(0, 2):
        print apply_training_step(session=sess, inputs=np.ones(shape=(5 * input_block_size, 1, input_dimensions)),
                                  input_block_size=input_block_size, targets=[1, 2, 1, 2, 1, 2],
                                  transducer_max_width=2)

    save_model_for_inference(sess, dir + '/model_save/model_test1')


with tf.Session() as sess2:
    inference = InferenceManager(session=sess2, beam_search=False, path=dir+'/model_save/model_test1',
                                 transducer_width=2)
    inference.run_inference(sess2, dir + '/model_save/model_test1',
                            full_inputs=np.ones(shape=(5 * input_block_size, 1, input_dimensions)))

