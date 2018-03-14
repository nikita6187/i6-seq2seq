import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.layers import core as layers_core
import numpy as np
import copy
import random
import cPickle
import os
import time

# Implementation of the "A Neural Transducer" paper, Navdeep Jaitly et. al (2015): https://arxiv.org/abs/1511.04868

# NOTE: Time major

# TODO: teacher forcing         [p]
# TODO: variable batch size     [p]
# TODO: distributed alignment
# TODO: inference update

# TODO: documentation


# ---------------- Constants Manager ----------------------------
class ConstantsManager(object):
    def __init__(self, input_dimensions, input_embedding_size, inputs_embedded, encoder_hidden_units,
                 transducer_hidden_units, vocab_ids, input_block_size, beam_width, encoder_hidden_layers,
                 transducer_max_width, path_to_model, path_to_inputs, path_to_targets, path_to_alignments,
                 path_to_cons_manager, amount_of_aligners, device_to_run, device_soft_placement,
                 debug_devices):
        assert transducer_hidden_units == 2 * encoder_hidden_units, 'Transducer has to have 2 times the amount ' \
                                                                    'of the encoder of units'
        # Vocab vars
        self.vocab_ids = vocab_ids
        self.E_SYMBOL = len(self.vocab_ids)
        self.vocab_ids.append('E_SYMBOL')
        self.GO_SYMBOL = len(self.vocab_ids)
        self.vocab_ids.append('GO_SYMBOL')
        self.PAD = len(self.vocab_ids)
        self.vocab_ids.append('PAD')
        self.vocab_size = len(self.vocab_ids)
        # Transducer vars
        self.input_dimensions = input_dimensions
        self.input_embedding_size = input_embedding_size
        self.inputs_embedded = inputs_embedded
        self.encoder_hidden_units = encoder_hidden_units
        self.transducer_hidden_units = transducer_hidden_units
        self.input_block_size = input_block_size
        self.beam_width = beam_width
        self.log_prob_init_value = 0
        self.encoder_hidden_layers = encoder_hidden_layers
        self.transducer_max_width = transducer_max_width
        # Path vars
        self.path_to_model = path_to_model
        self.path_to_inputs = path_to_inputs
        self.path_to_alignments = path_to_alignments
        self.path_to_targets = path_to_targets
        self.path_to_cons_manager = path_to_cons_manager
        # Alignment managing
        self.amount_of_aligners = amount_of_aligners
        self.device_to_run = device_to_run
        self.device_soft_placement = device_soft_placement
        self.debug_devices = debug_devices


    # TODO: add lookup function for vocab


# ---------------- Helper classes -------------------------------

class Alignment(object):
    def __init__(self, cons_manager):
        self.alignment_position = (0, 1)  # x = position in target (y~), y = block index, both start at 1
        self.log_prob = cons_manager.log_prob_init_value  # The sum log prob of this alignment over the target indices
        self.alignment_locations = []  # At which indices in the target output we need to insert <e>
        self.last_state_transducer = np.zeros(shape=(2, 1, cons_manager.transducer_hidden_units))  # Transducer state
        self.cons_manager = cons_manager

    def __compute_sum_probabilities(self, transducer_outputs, targets, transducer_amount_outputs):
        def get_prob_at_timestep(timestep):
            return np.log(transducer_outputs[timestep][0][targets[start_index + timestep]])

        start_index = self.alignment_position[0] - transducer_amount_outputs  # The current position of this alignment
        prob = self.cons_manager.log_prob_init_value
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
        self.log_prob += self.__compute_sum_probabilities(transducer_outputs, targets, transducer_amount_outputs)
        self.last_state_transducer = new_transducer_state


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class DataManager(object):
    def __init__(self, cons_manager, full_inputs, full_targets, model, session, online_alignments):
        """
        Loads the data manager
        :param cons_manager:
        :param full_inputs: Of shape [max_input_time, amount, ...] (Time major)
        :param full_targets: Of shape [amount, max_output_time, ...] (Batch major)
        :param model: The model object
        """
        assert full_inputs.shape[1] == len(full_targets), 'Input batch size not equal to target batch size!'

        self.data_dic = {}  # Key is the input as string, each entry of shape (input, targets, alignment)
        self.cons_manager = cons_manager
        self.inputs = full_inputs
        self.targets = full_targets
        self.model = model
        self.session = session
        self.online_alignments = online_alignments

        # Save inputs, targets, model & cons_manager
        np.save(self.cons_manager.path_to_inputs, self.inputs)
        np.save(self.cons_manager.path_to_targets, np.asarray(self.targets))
        model.save_model_for_inference(session, path_name=self.cons_manager.path_to_model)
        cons_man_file = open(self.cons_manager.path_to_cons_manager, 'wb')
        cPickle.dump(self.cons_manager, cons_man_file)
        cons_man_file.close()

        # Init the data dictionary
        for sample_id in range(full_inputs.shape[1]):
            self.data_dic[self.inputs[:, sample_id, :].tostring()] = (np.reshape(self.inputs[:, sample_id, :], newshape=(-1, 1, self.cons_manager.input_dimensions)),
                                                                      self.targets[sample_id],
                                                                      None)

    def run_new_alignments(self):
        print 'Loading in new alignments'
        # Save model and run new alignments
        self.model.save_model_for_inference(self.session, path_name=self.cons_manager.path_to_model)
        os.system('python ./neural_transducer_helpers.py ' + str(self.cons_manager.path_to_cons_manager))

        # Waits until new alignments are there, so the alignments file has changed
        # TODO: something more elegant

        # Load in new alignments
        new_al_file = open(self.cons_manager.path_to_alignments, 'rb')
        new_alignments = cPickle.load(new_al_file)
        new_al_file.close()

        # Apply new alignments to internal dictionary
        for input_key in new_alignments:
            self.data_dic[input_key] = (self.data_dic[input_key][0], self.data_dic[input_key][1], new_alignments[input_key])
        print 'New alignments loaded'

    def get_new_sample(self, inputs):
        if self.online_alignments is True:
            (inp, targ, _) = self.data_dic[inputs]
            al = self.model.get_alignment(session=self.session, inputs=inp, targets=targ,
                                          input_block_size=self.cons_manager.input_block_size,
                                          transducer_max_width=self.cons_manager.transducer_max_width)
        else:
            # Skip None alignments
            # TODO: some error here
            (inp, targ, al) = self.data_dic[inputs]
            while al is None:
                (inp, targ, al) = self.data_dic[inputs]
        return inp, targ, al

    def get_new_random_sample(self):
        key = random.choice(self.data_dic.keys())
        return self.get_new_sample(key)

    def set_online_alignment(self, mode):
        """
        Sets either to use pre-calculated alignments or to calculate them on the fly (uses more processing time,
        but more precise).
        :param mode: True/False, True for online alignments.
        :return:
        """
        self.online_alignments = mode


# ----------------- Model ---------------------------------------

class Model(object):

    def __init__(self, cons_manager):

        self.var_list = []
        self.cons_manager = cons_manager

        self.max_blocks, self.inputs_full_raw, self.transducer_list_outputs, self.start_block, \
            self.encoder_hidden_init_fw, self.encoder_hidden_init_bw,\
            self.trans_hidden_init, self.teacher_forcing_targets, self.inference_mode, self.logits, \
            self.encoder_hidden_state_new_fw, self.encoder_hidden_state_new_bw, \
            self.transducer_hidden_state_new, self.train_saver = self.build_full_transducer()

        self.targets, self.train_op, self.loss = self.build_training_step()

    def build_full_transducer(self):
        with tf.variable_scope('transducer_training'):

            embeddings = tf.Variable(tf.random_uniform([self.cons_manager.vocab_size,
                                                        self.cons_manager.input_embedding_size], -1.0, 1.0),
                                     dtype=tf.float32,
                                     name='embedding')
            # Inputs
            max_blocks = tf.placeholder(dtype=tf.int32, name='max_blocks')  # total amount of blocks to go through
            if self.cons_manager.inputs_embedded is True:
                input_type = tf.float32
            else:
                input_type = tf.int32
            inputs_full_raw = tf.placeholder(shape=(None, None,
                                                    self.cons_manager.input_dimensions), dtype=input_type,
                                             name='inputs_full_raw')  # shape [max_time, batch_size, input_dims]
            transducer_list_outputs = tf.placeholder(shape=(None, None), dtype=tf.int32,
                                                     name='transducer_list_outputs')  # amount to output per block, [max_blocks, batch_size]
            start_block = tf.placeholder(dtype=tf.int32, name='transducer_start_block')  # where to start the input

            encoder_hidden_init_fw = tf.placeholder(shape=(self.cons_manager.encoder_hidden_layers,
                                                    2,
                                                    None,
                                                    self.cons_manager.encoder_hidden_units), dtype=tf.float32,
                                                    name='encoder_hidden_init_fw')
            encoder_hidden_init_bw = tf.placeholder(shape=(self.cons_manager.encoder_hidden_layers,
                                                           2,
                                                           None,
                                                           self.cons_manager.encoder_hidden_units), dtype=tf.float32,
                                                    name='encoder_hidden_init_bw')

            trans_hidden_init = tf.placeholder(shape=(2, None,
                                                      self.cons_manager.transducer_hidden_units), dtype=tf.float32,
                                               name='trans_hidden_init')

            # Only has to contain data if in training
            # should be padded (PAD) so that each example has the same amount of target inputs per transducer block
            # [max_time, batch_size]
            teacher_forcing_targets = tf.placeholder(shape=(None, None), dtype=tf.int32,
                                                     name='teacher_forcing_targets')
            inference_mode = tf.placeholder(shape=(),
                                            dtype=tf.float32, name='inference_mode')  # Set 1.0 for inference, <1.0 for training
            # Get batch size
            batch_size = tf.shape(inputs_full_raw)[1]


            # Temporary constants, maybe changed during inference
            end_symbol = tf.get_variable(name='end_symbol',
                                         initializer=tf.constant_initializer(self.cons_manager.vocab_size),
                                         shape=(), dtype=tf.int32)

            # Process teacher forcing targets
            teacher_forcing_targets_emb = tf.nn.embedding_lookup(embeddings, teacher_forcing_targets)

            # Turn inputs into tensor which is easily readable
            inputs_full = tf.reshape(inputs_full_raw, shape=[-1, self.cons_manager.input_block_size,
                                                             batch_size,
                                                             self.cons_manager.input_dimensions])

            #inputs_full = tf.Print(inputs_full, [inputs_full], message='Inputs', summarize=10)

            # Outputs
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_blocks)
            init_state = (start_block, outputs_ta, encoder_hidden_init_fw, encoder_hidden_init_bw, trans_hidden_init,
                          0)

            # Initiate cells
            cell = []
            for i in range(self.cons_manager.encoder_hidden_layers):
                cell.append(tf.contrib.rnn.LSTMCell(self.cons_manager.encoder_hidden_units, state_is_tuple=True))
            encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)
            cell = []
            for i in range(self.cons_manager.encoder_hidden_layers):
                cell.append(tf.contrib.rnn.LSTMCell(self.cons_manager.encoder_hidden_units, state_is_tuple=True))
            encoder_cell_bw = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)

            transducer_cell = tf.contrib.rnn.LSTMCell(self.cons_manager.transducer_hidden_units)

            def cond(current_block, outputs_int, encoder_hidden_fw, encoder_hidden_bw, trans_hidden, total_output):
                return current_block < start_block + max_blocks

            def body(current_block, outputs_int, encoder_hidden_fw, encoder_hidden_bw, trans_hidden, total_output):

                # --------------------- ENCODER ----------------------------------------------------------------------
                encoder_inputs = inputs_full[current_block]

                if self.cons_manager.inputs_embedded is True:
                    encoder_inputs_embedded = encoder_inputs
                else:
                    encoder_inputs = tf.reshape(encoder_inputs, shape=[-1, batch_size])
                    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
                    # TODO: see if encoder_inputs_embedded is time major

                # Build model
                # Process encoder state

                l_fw = tf.unstack(encoder_hidden_fw, axis=0)
                encoder_hidden_state_fw = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(l_fw[idx][0], l_fw[idx][1])
                     for idx in range(self.cons_manager.encoder_hidden_layers)]
                )

                l_bw = tf.unstack(encoder_hidden_bw, axis=0)
                encoder_hidden_state_bw = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(l_bw[idx][0], l_bw[idx][1])
                     for idx in range(self.cons_manager.encoder_hidden_layers)]
                )
                #   encoder_outputs: [max_time, batch_size, num_units]
                ((encoder_outputs_fw, encoder_outputs_bw), (encoder_hidden_state_new_fw, encoder_hidden_state_new_bw)) = \
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw, cell_bw=encoder_cell_bw,
                                                    inputs=encoder_inputs_embedded,
                                                    time_major=True,
                                                    dtype=tf.float32,
                                                    initial_state_fw=encoder_hidden_state_fw,
                                                    initial_state_bw=encoder_hidden_state_bw)

                encoder_outputs = tf.concat([encoder_outputs_fw, encoder_outputs_bw], 2)

                # Modify output of encoder_hidden_state_new so that it can be fed back in again without problems.
                encoder_hidden_state_new_fw = tf.concat(
                    [tf.concat([ehs.c, ehs.h], axis=0) for ehs in encoder_hidden_state_new_fw],
                    axis=0)
                encoder_hidden_state_new_fw = tf.reshape(encoder_hidden_state_new_fw,
                                                      shape=[self.cons_manager.encoder_hidden_layers,
                                                             2,
                                                             batch_size,
                                                             self.cons_manager.encoder_hidden_units])

                encoder_hidden_state_new_bw = tf.concat(
                    [tf.concat([ehs.c, ehs.h], axis=0) for ehs in encoder_hidden_state_new_bw],
                    axis=0)
                encoder_hidden_state_new_bw = tf.reshape(encoder_hidden_state_new_bw,
                                                         shape=[self.cons_manager.encoder_hidden_layers,
                                                                2,
                                                                batch_size,
                                                                self.cons_manager.encoder_hidden_units])

                # --------------------- TRANSDUCER --------------------------------------------------------------------
                # Each transducer block runs for the max transducer outputs in its respective block

                encoder_raw_outputs = encoder_outputs
                # Save/load the state as one tensor, use top encoder layer state as init if this is the first block
                trans_hidden_state = tf.cond(current_block > 0,
                                             lambda: trans_hidden,
                                             lambda: tf.concat([encoder_hidden_state_new_fw[-1], encoder_hidden_state_new_bw[-1]], 2))  # TODO: see if index is '0' or '-1'
                transducer_amount_outputs = transducer_list_outputs[current_block - start_block]
                transducer_max_output = tf.reduce_max(transducer_amount_outputs)

                # Model building
                helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=teacher_forcing_targets_emb[total_output:total_output + transducer_max_output],  # Get the current target inputs
                    sequence_length=transducer_amount_outputs,
                    embedding=embeddings,
                    sampling_probability=inference_mode,
                    time_major=True
                )

                attention_states = tf.transpose(encoder_raw_outputs,
                                                [1, 0, 2])  # attention_states: [batch_size, max_time, num_units]

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.cons_manager.encoder_hidden_units * 2, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    transducer_cell,
                    attention_mechanism,
                    attention_layer_size=self.cons_manager.transducer_hidden_units)

                projection_layer = layers_core.Dense(self.cons_manager.vocab_size, use_bias=False)

                # Build previous state
                trans_hidden_c, trans_hidden_h = tf.split(trans_hidden_state, num_or_size_splits=2, axis=0)
                trans_hidden_c = tf.reshape(trans_hidden_c, shape=[-1, self.cons_manager.transducer_hidden_units])
                trans_hidden_h = tf.reshape(trans_hidden_h, shape=[-1, self.cons_manager.transducer_hidden_units])
                trans_hidden_state_t = LSTMStateTuple(trans_hidden_c, trans_hidden_h)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper,
                    decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=trans_hidden_state_t),
                    output_layer=projection_layer)
                outputs, transducer_hidden_state_new, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                            output_time_major=True,
                                                                                            maximum_iterations=transducer_max_output)
                logits = outputs.rnn_output  # logits of shape [max_time,batch_size,vocab_size]
                decoder_prediction = outputs.sample_id  # For debugging

                # Modify output of transducer_hidden_state_new so that it can be fed back in again without problems.

                transducer_hidden_state_new = tf.concat(
                    [transducer_hidden_state_new[0].c, transducer_hidden_state_new[0].h],
                    axis=0)
                transducer_hidden_state_new = tf.reshape(transducer_hidden_state_new,
                                                         shape=[2, -1, self.cons_manager.transducer_hidden_units])

                # Note the outputs
                outputs_int = outputs_int.write(current_block - start_block, logits)

                return current_block + 1, outputs_int, encoder_hidden_state_new_fw, encoder_hidden_state_new_bw, \
                    transducer_hidden_state_new, total_output + transducer_max_output

            _, outputs_final, encoder_hidden_state_new_fw, encoder_hidden_state_new_bw, \
                transducer_hidden_state_new, _ = tf.while_loop(cond, body, init_state, parallel_iterations=1)

            # Process outputs
            logits = outputs_final.concat()  # And now its [max_output_time, batch_size, vocab]

            # For loading the model later on
            logits = tf.identity(logits, name='logits')
            encoder_hidden_state_new_fw = tf.identity(encoder_hidden_state_new_fw, name='encoder_hidden_state_new_fw')
            encoder_hidden_state_new_bw = tf.identity(encoder_hidden_state_new_bw, name='encoder_hidden_state_new_bw')
            transducer_hidden_state_new = tf.identity(transducer_hidden_state_new, name='transducer_hidden_state_new')

        train_saver = tf.train.Saver()  # For now save everything

        return max_blocks, inputs_full_raw, transducer_list_outputs, start_block, encoder_hidden_init_fw, \
            encoder_hidden_init_bw, trans_hidden_init, teacher_forcing_targets, inference_mode, \
            logits, encoder_hidden_state_new_fw, encoder_hidden_state_new_bw, transducer_hidden_state_new, train_saver

    def build_training_step(self):
        # All targets should be the same lengths, and be adjusted for this in preprocessing
        # Of shape [max_time, batch_size]
        targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
        targets_one_hot = tf.one_hot(targets, depth=self.cons_manager.vocab_size, dtype=tf.float32, name='targets_one_hot')

        targets_one_hot = tf.Print(targets_one_hot, [targets], message='Targets: ', summarize=100)
        targets_one_hot = tf.Print(targets_one_hot, [tf.argmax(self.logits, axis=2)], message='Argmax: ', summarize=100)

        self.logits = tf.identity(self.logits, name='training_logits')
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot, logits=self.logits)

        loss = tf.reduce_mean(stepwise_cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)
        return targets, train_op, loss

    def __create_loading_dic(self, list_vars, old_name, new_name):
        dic = {}
        for var in list_vars:
            dic[var.name] = var.name.replace(old_name, new_name)
        return dic

    # TODO: This is identical to the function in neural_transducer_helpers.py:AlignerWorker, used for hybrid mode
    def get_alignment(self, session, inputs, targets, input_block_size, transducer_max_width):
        """
        Finds the alignment of the target sequence to the actual output.
        :param session: The current session.
        :param inputs: The complete inputs for the encoder of shape [max_time, 1, input_dimensions], note padding if needed
        :param targets: The target sequence of shape [time] where each enty is an index.
        :param input_block_size: The width of one encoder block.
        :param transducer_max_width: The max width of one transducer block.
        :return: Returns a list of indices where <e>'s need to be inserted into the target sequence. (see paper)
        """
        model = self
        self.full_time_needed_transducer = 0

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
                :param encoder_state: Tuple containing (encoder_state_fw, encoder_state_bw). Each state of size [encoder
                layers, 2, 1, encoder_hidden_units]
                :param transducer_state: The last transducer state as [2, 1, transducer_hidden_units] tensor.
                :param transducer_width: The amount of outputs the transducer should produce.
                :return: transducer outputs [max_output_time, 1, vocab], transducer_state [2, 1, transducer_hidden_units],
                encoder_state [2, 1, encoder_hidden_units]
                """
                teacher_targets_empty = np.ones([transducer_width, 1]) * self.cons_manager.GO_SYMBOL  # Only use go, rest is greedy

                temp_init_time = time.time()

                logits, trans_state, enc_state_fw, enc_state_bw = session.run([model.logits, model.transducer_hidden_state_new,
                                                              model.encoder_hidden_state_new_fw, model.encoder_hidden_state_new_bw],
                                                             feed_dict={
                                                                 model.inputs_full_raw: inputs_full,
                                                                 model.max_blocks: 1,
                                                                 model.transducer_list_outputs: [[transducer_width]],
                                                                 model.start_block: block_index - 1,
                                                                 model.encoder_hidden_init_fw: encoder_state[0],
                                                                 model.encoder_hidden_init_bw: encoder_state[1],
                                                                 model.trans_hidden_init: transducer_state,
                                                                 model.inference_mode: 1.0,
                                                                 model.teacher_forcing_targets: teacher_targets_empty,
                                                             })
                model.full_time_needed_transducer += time.time() - temp_init_time

                # apply softmax on the outputs
                trans_out = softmax(logits, axis=2)

                return trans_out, trans_state, (enc_state_fw, enc_state_bw)

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
                                                                                    transducer_width=new_alignment_width + 1)
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
        current_alignments = [Alignment(cons_manager=self.cons_manager)]
        last_encoder_state = (np.zeros(shape=(self.cons_manager.encoder_hidden_layers, 2, 1, self.cons_manager.encoder_hidden_units)),
                              np.zeros(shape=(self.cons_manager.encoder_hidden_layers, 2, 1, self.cons_manager.encoder_hidden_units)))

        # Do assertions to check whether everything was correctly set up.
        assert inputs.shape[0] % input_block_size == 0, \
            'Input shape not corresponding to input block size (add padding or see if batch first).'
        assert inputs.shape[
                   2] == self.cons_manager.input_dimensions, 'Input dimension [2] not corresponding to specified input dimension.'
        assert inputs.shape[1] == 1, 'Batch size needs to be one.'
        assert transducer_max_width * amount_of_input_blocks >= len(
            targets), 'transducer_max_width to small for targets'

        for block in range(current_block_index, amount_of_input_blocks + 1):
            # Run all blocks
            current_alignments, last_encoder_state = run_new_block(session=session, full_inputs=inputs,
                                                                   previous_alignments=current_alignments,
                                                                   block_index=block,
                                                                   transducer_max_width=transducer_max_width,
                                                                   targets=targets, total_blocks=amount_of_input_blocks,
                                                                   last_encoder_state=last_encoder_state)
            #print 'Size of alignments: ' + str(float(asizeof.asizeof(current_alignments))/(1024 * 1024))

        # Select first alignment if we have multiple with the same log prob (happens with ~1% probability in training)

        print 'Full time needed for transducer: ' + str(self.full_time_needed_transducer)

        return current_alignments[0].alignment_locations

    def apply_training_step(self, session, batch_size, data_manager):
        """
        Applies a training step to the transducer model. This method can be called multiple times from e.g. a loop.
        :param session: The current session.
        :param inputs: The full inputs. Shape: [max_time, batch_size, input_dimensions]
        :param targets: The full targets. Shape: [batch_size, time]. Each entry is an index. Use lists.
        All targets have to have the same length. (tl;dr: A list containing a list for each target, same lengths).
        :param input_block_size: The block width for the inputs.
        :param transducer_max_width: The max width for the transducer. Not including the output symbol <e>
        :param training_steps_per_alignment: The amount of times to repeat the training step whilst caching the same
        alignment.
        :return: Average loss of this training step.
        """

        # Get vars
        alignments = []
        inputs = []
        targets = []

        # Get batch size amount of data
        for i in range(batch_size):
            (temp_inputs, target, alignment) = data_manager.get_new_random_sample()

            alignments.append(alignment)
            targets.append(list(target))
            inputs.append(temp_inputs)

        inputs = np.concatenate(inputs, axis=1)

        print 'Alignment: ' + str(alignments)

        # Set vars
        teacher_forcing = []
        lengths = []
        max_lengths = [0] * len(alignments[0])
        batch_size = inputs.shape[1]

        # First calculate (max) lengths for all sequences
        for batch_index in range(batch_size):
            alignment = alignments[batch_index]

            # Calc temp true & max lengths for each transducer block
            lengths_temp = []
            alignment.insert(0, 0)  # This is so that the length calculation is done correctly
            for i in range(1, len(alignment)):
                lengths_temp.append(alignment[i] - alignment[i - 1] + 1)
                max_lengths[i-1] = max(max_lengths[i-1], lengths_temp[i-1])  # For later use; how long each block is
            del alignment[0]  # Remove alignment index that we added
            lengths.append(lengths_temp)

        # Next modify so that each sequence is of equal length in each transducer block & targets have alignments
        for batch_index in range(inputs.shape[1]):
            alignment = alignments[batch_index]

            # Modify targets so that it has the appropriate alignment
            offset = 0
            for e in alignment:
                targets[batch_index].insert(e + offset, self.cons_manager.E_SYMBOL)
                offset += 1

            # Modify so that all targets have same lengths in each transducer using PAD
            offset = 0
            for i in range(len(alignment)):
                for app in range(max_lengths[i] - lengths[batch_index][i]):
                    targets[batch_index].insert(offset + lengths[batch_index][i], self.cons_manager.PAD)
                offset += max_lengths[i]

            # Modify targets for teacher forcing
            teacher_forcing_temp = list(targets[batch_index])
            teacher_forcing_temp.insert(0, self.cons_manager.GO_SYMBOL)
            teacher_forcing_temp.pop(len(teacher_forcing_temp) - 1)
            for i in range(len(teacher_forcing_temp)):
                if teacher_forcing_temp[i] == self.cons_manager.E_SYMBOL \
                        and targets[batch_index][i] != self.cons_manager.PAD:
                    teacher_forcing_temp[i] = self.cons_manager.GO_SYMBOL

                if i + 1 < len(teacher_forcing_temp) and \
                        teacher_forcing_temp[i] == self.cons_manager.PAD and \
                        teacher_forcing_temp[i + 1] != self.cons_manager.PAD:
                    teacher_forcing_temp[i] = self.cons_manager.GO_SYMBOL

            teacher_forcing.append(teacher_forcing_temp)

        # Process targets back to time major
        targets = np.asarray(targets)
        targets = np.transpose(targets, axes=[1, 0])

        # See that teacher forcing are of correct format
        teacher_forcing = np.asarray(teacher_forcing)
        teacher_forcing = np.transpose(teacher_forcing, axes=[1, 0])

        # Process lengths
        lengths = np.asarray(lengths)
        lengths = np.transpose(lengths, axes=[1, 0])

        # Init values
        encoder_hidden_init = (np.zeros(shape=(self.cons_manager.encoder_hidden_layers, 2, batch_size, self.cons_manager.encoder_hidden_units)),
                               np.zeros(shape=(self.cons_manager.encoder_hidden_layers, 2, batch_size, self.cons_manager.encoder_hidden_units)))
        trans_hidden_init = np.zeros(shape=(2, batch_size, self.cons_manager.transducer_hidden_units))

        # Run training step
        _, loss = session.run([self.train_op, self.loss], feed_dict={
            self.max_blocks: len(lengths),
            self.inputs_full_raw: inputs,
            self.transducer_list_outputs: lengths,
            self.targets: targets,
            self.start_block: 0,
            self.encoder_hidden_init_fw: encoder_hidden_init[0],
            self.encoder_hidden_init_bw: encoder_hidden_init[1],
            self.trans_hidden_init: trans_hidden_init,
            self.inference_mode: 0.0,
            self.teacher_forcing_targets: teacher_forcing,
        })

        return loss

    def save_model_for_inference(self, session, path_name):
        self.train_saver.save(session, path_name)
        print 'Model saved to ' + str(path_name)


class InferenceManager(object):

    def __init__(self, cons_manager, session):
        self.cons_manager = cons_manager
        # Init the interface for model loading
        self.max_blocks = self.inputs_full_raw = self.transducer_list_outputs = self.start_block = \
            self.encoder_hidden_init_fw = self.encoder_hidden_init_bw = \
            self.trans_hidden_init = self.teacher_forcing_targets = self.inference_mode = self.logits = \
            self.encoder_hidden_state_new_fw = self.encoder_hidden_state_new_bw = \
            self.transducer_hidden_state_new = None

    def build_greedy_inference(self, path, session):
        # Restore graph
        saver = tf.train.import_meta_graph(path + '.meta')
        saver.restore(session, path)
        # Setup constants
        graph = tf.get_default_graph()
        # self.end_symbol = graph.get_tensor_by_name(name='transducer_training/end_symbol:0')
        # Get inputs
        self.max_blocks = graph.get_tensor_by_name(name='transducer_training/max_blocks:0')
        self.inputs_full_raw = graph.get_tensor_by_name(name='transducer_training/inputs_full_raw:0')
        self.transducer_list_outputs = graph.get_tensor_by_name(name='transducer_training/transducer_list_outputs:0')
        self.start_block = graph.get_tensor_by_name(name='transducer_training/transducer_start_block:0')
        self.encoder_hidden_init_fw = graph.get_tensor_by_name(name='transducer_training/encoder_hidden_init_fw:0')
        self.encoder_hidden_init_bw = graph.get_tensor_by_name(name='transducer_training/encoder_hidden_init_bw:0')
        self.trans_hidden_init = graph.get_tensor_by_name(name='transducer_training/trans_hidden_init:0')
        self.teacher_forcing_targets = graph.get_tensor_by_name(name='transducer_training/teacher_forcing_targets:0')
        self.inference_mode = graph.get_tensor_by_name(name='transducer_training/inference_mode:0')
        # Get return ops
        self.logits = graph.get_operation_by_name(name='transducer_training/logits').outputs[0]
        self.encoder_hidden_state_new_fw = \
        graph.get_operation_by_name(name='transducer_training/encoder_hidden_state_new_fw').outputs[0]
        self.encoder_hidden_state_new_bw = \
        graph.get_operation_by_name(name='transducer_training/encoder_hidden_state_new_bw').outputs[0]
        self.transducer_hidden_state_new = \
        graph.get_operation_by_name(name='transducer_training/transducer_hidden_state_new').outputs[0]

        return max_blocks, inputs_full_raw, transducer_list_outputs, start_block, encoder_hidden_init, \
            trans_hidden_init, logits, encoder_hidden_state_new, transducer_hidden_state_new

    def run_inference(self, session, full_inputs, clean_e):
        # Can only process 1 sequence at a time

        # TODO: build this so running a block is correct
        # TODO: create beam search for optimimum next block phase
        # TODO: test & debug

        def run_greedy_transducer_block(session, full_inputs, current_block, encoder_init_state, transducer_init_state):
            logits, encoder_new_state, transducer_new_state = \
                session.run([self.logits, self.encoder_hidden_state_new,
                             self.transducer_hidden_state_new], feed_dict={
                    self.inputs_full_raw: full_inputs,
                    self.max_blocks: 1,
                    self.transducer_list_outputs: [self.transducer_width+1],  # +1 due to <e> not being in trans width
                    self.start_block: current_block,
                    self.encoder_hidden_init: encoder_init_state,
                    self.trans_hidden_init: transducer_init_state
                })
            # TODO test softmax the logits
            logits = softmax(logits, axis=2)
            return logits, encoder_new_state, transducer_new_state

        # Meta parameters
        amount_of_input_blocks = int(np.ceil(full_inputs.shape[0] / self.cons_manager.input_block_size))
        # Init encoder/decoder states
        last_encoder_state = np.zeros(shape=(2, 1, self.cons_manager.encoder_hidden_units))
        last_transducer_state = np.zeros(shape=(2, 1, self.cons_manager.transducer_hidden_units))
        logits = []

        for current_input_block in range(0, amount_of_input_blocks):
                # TODO: here
                new_logits, last_encoder_state, last_transducer_state = \
                    run_greedy_transducer_block(session=session,
                                                full_inputs=full_inputs,
                                                current_block=current_input_block,
                                                encoder_init_state=last_encoder_state,
                                                transducer_init_state=last_transducer_state)
                logits.append(new_logits)

        # Post process logits into one np array and transform into list of ids
        logit_arr = np.concatenate(logits, axis=0)  # Is now of shape [max_time, 1, vocab_size]
        predict_id = np.argmax(logit_arr, axis=2)
        predict_id = np.reshape(predict_id, newshape=(-1))
        predict_id = predict_id.tolist()

        def lookup(i):
            return self.cons_manager.vocab_ids[i]
        predicted_chars = map(lookup, predict_id)
        if clean_e is True:
            predict_id = [i for i in predict_id if i != self.cons_manager.E_SYMBOL]

        return predict_id, predicted_chars


# Visualization
"""
constants_manager = ConstantsManager(input_dimensions=1, input_embedding_size=11, inputs_embedded=False,
                                     encoder_hidden_units=100, transducer_hidden_units=200, vocab_ids=[0, 1, 2],
                                     input_block_size=1, beam_width=5, encoder_hidden_layers=3)
model = Model(cons_manager=constants_manager)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("/home/nikita/Desktop", sess.graph_def)
    sess.run(tf.global_variables_initializer())
    writer.flush()
    writer.close()
"""

