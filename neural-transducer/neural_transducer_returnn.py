import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMStateTuple
import copy
import time


class NeuralTransducerLayer(_ConcatInputLayer):
    """
    Performs a neural transducer based on the paper "A Neural Transducer": https://arxiv.org/abs/1511.04868.
    NOTE: Requires that the targets be modified, for example using the [INSERT HERE CORRECT NAME] aligner.
    """
    layer_class = "neural_transducer_layer"

    def __init__(self, ):
        " docstring, document the args! "

        # TODO: set everything correctly here
        super(NeuralTransducerLayer, self).__init__(**kwargs)

        self.output.placeholder = self.build_full_transducer()
        self.output.size_placeholder = [max_output_time, batch_size, vocab_size], \
                                       [2, batch_size, transducer_hidden_units]

    def build_full_transducer(self, transducer_hidden_units, embeddings, batch_size, vocab_size, input_block_size,
                              transducer_list_outputs, max_blocks, trans_hidden_init, teacher_forcing_targets,
                              inference_mode, encoder_outputs):

        # TODO: Get the following variables
        # - transducer_hidden_units (int32, static
        # - batch_size (int32, static)
        # - vocab_size (int32, static)
        # - input_block_size (int32, static)

        # - embeddings [vocab_size, embedding_size]
        # - transducer_list_outputs amount to output per block, [max_blocks, batch_size]
        # - max_blocks (int32) total amount of blocks to go through
        # - trans_hidden_init [2, batch_size, transducer_hidden_units] transducer init state
        # - teacher_forcing_targets [max_time, batch_size] Only has to contain data if in training,
        #                   should be padded (PAD) so that each example has the same amount of target
        #                   inputs per transducer block
        # - inference_mode (float32) 1 for inference (no teacher forcing) or 0 (or in between)
        # - encoder_outputs [max_times, batch_size, encoder_hidden]

        with tf.variable_scope('transducer_training'):

            # Process teacher forcing targets
            teacher_forcing_targets_emb = tf.nn.embedding_lookup(embeddings, teacher_forcing_targets)

            # Outputs
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_blocks, infer_shape=False)
            init_state = (0, outputs_ta, trans_hidden_init, 0)

            transducer_cell = tf.contrib.rnn.LSTMCell(transducer_hidden_units)

            def cond(current_block, outputs_int, trans_hidden, total_output):
                return current_block < max_blocks

            def body(current_block, outputs_int, trans_hidden, total_output):

                # --------------------- TRANSDUCER --------------------------------------------------------------------
                # Each transducer block runs for the max transducer outputs in its respective block

                # TODO: get encoder outputs

                encoder_raw_outputs = encoder_outputs[input_block_size * current_block:input_block_size * (current_block + 1)]

                # Save/load the state as one tensor, use top encoder layer state as init if this is the first block
                trans_hidden_state = trans_hidden
                transducer_amount_outputs = transducer_list_outputs[current_block]
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
                    transducer_hidden_units, attention_states)

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
                                                         shape=[2, -1, transducer_hidden_units])

                # Note the outputs
                outputs_int = outputs_int.write(current_block, logits)

                return current_block + 1, outputs_int, \
                    transducer_hidden_state_new, total_output + transducer_max_output

            _, outputs_final, transducer_hidden_state_new, _ = tf.while_loop(cond, body, init_state,
                                                                             parallel_iterations=1)

            # Process outputs
            logits = outputs_final.concat()  # And now its [max_output_time, batch_size, vocab]

            # For loading the model later on
            logits = tf.identity(logits, name='logits')
            transducer_hidden_state_new = tf.identity(transducer_hidden_state_new, name='transducer_hidden_state_new')

        return logits, transducer_hidden_state_new

    @classmethod
    def get_out_data_from_opts(cls, **kwargs):
        " This is supposed to return a :class:`Data` instance as a template, given the arguments. "
        # TODO: make this correct
        # example, just the same as the input:
        return get_concat_sources_data_template(kwargs["sources"], name="%s_output" % kwargs["name"])


class Alignment(object):

    def __init__(self, transducer_hidden_units, E_SYMBOL):
        self.alignment_position = (0, 1)  # x = position in target (y~), y = block index, both start at 1
        self.log_prob = 0  # The sum log prob of this alignment over the target indices
        self.alignment_locations = []  # At which indices in the target output we need to insert <e>
        self.last_state_transducer = np.zeros(
            shape=(2, 1, transducer_hidden_units))  # Transducer state
        self.E_SYMBOL = E_SYMBOL

    def __compute_sum_probabilities(self, transducer_outputs, targets, transducer_amount_outputs):
        def get_prob_at_timestep(timestep):
            if timestep + start_index < len(targets):
                # For normal operations
                # print 'H'
                # print np.log(transducer_outputs[timestep][0][targets[start_index + timestep]])
                # print np.log(transducer_outputs[timestep][0][self.cons_manager.E_SYMBOL])
                # print targets[start_index + timestep]
                return np.log(transducer_outputs[timestep][0][targets[start_index + timestep]])
            else:
                # For last timestep, so the <e> symbol
                return np.log(transducer_outputs[timestep][0][self.E_SYMBOL])

        # print transducer_outputs
        start_index = self.alignment_position[
                          0] - transducer_amount_outputs  # The current position of this alignment
        prob = 0
        for i in range(0,
                       transducer_amount_outputs + 1):  # Do not include e symbol in calculation, +1 due to last symbol
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

class NeuralTransducerAligner():

    def __init__(self):

    def get_alignment(self, session, encoder_outputs, targets, input_block_size, transducer_max_width, transducer_hidden_units,
                      E_SYMBOL, GO_SYMBOL):

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

        def run_new_block(session, encoder_outputs, previous_alignments, block_index, transducer_max_width, targets,
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

            def run_transducer(session, encoder_outputs, encoder_state, transducer_state, transducer_width):
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
                """"""
                teacher_targets_empty = np.ones(
                    [transducer_width, 1]) * GO_SYMBOL  # Only use go, rest is greedy

                temp_init_time = time.time()
                # TODO: make this

                """
                logits, trans_state, enc_state_fw, enc_state_bw = session.run(
                    [model.logits, model.transducer_hidden_state_new,
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
                """
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
                                                                                    encoder_outputs=encoder_outputs,
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
        amount_of_input_blocks = int(np.ceil(encoder_outputs.shape[0] / input_block_size))
        current_block_index = 1
        current_alignments = [Alignment(transducer_hidden_units=transducer_hidden_units, E_SYMBOL=E_SYMBOL)]

        # Do assertions to check whether everything was correctly set up.
        assert transducer_max_width * amount_of_input_blocks >= len(
            targets), 'transducer_max_width to small for targets'

        for block in range(current_block_index, amount_of_input_blocks + 1):
            # Run all blocks
            current_alignments, last_encoder_state = run_new_block(session=session, encoder_outputs=encoder_outputs,
                                                                   previous_alignments=current_alignments,
                                                                   block_index=block,
                                                                   transducer_max_width=transducer_max_width,
                                                                   targets=targets, total_blocks=amount_of_input_blocks)
            # for alignment in current_alignments:
            # print str(alignment.alignment_locations) + ' ' + str(alignment.log_prob)

            # print 'Size of alignments: ' + str(float(asizeof.asizeof(current_alignments))/(1024 * 1024))

        print 'Full time needed for transducer: ' + str(self.full_time_needed_transducer)

        return current_alignments[0].alignment_locations

    def get_all_alignments(self, session, encoder_outputs, targets, input_block_size, transducer_max_width, transducer_hidden_units,
                      E_SYMBOL, GO_SYMBOL, PAD_SYMBOL):
        # encoder_outputs of shape [max_time, batch_size, encoder_hidden]
        # targets of shape [batch_size, max_target_time]

        # Using lists for now for easyness
        targets = targets.tolist()

        # Get vars
        alignments = []
        targets = []
        batch_size = encoder_outputs.shape[0]

        init_time = time.time()

        # Get batch size amount of data
        for i in range(batch_size):
            alignment = self.get_alignment(session=session, encoder_outputs=encoder_outputs[:,i,:],
                                                                  targets=targets[i],
                                                                  input_block_size=input_block_size,
                                                                  transducer_max_width=transducer_max_width,
                                                                  transducer_hidden_units=transducer_hidden_units,
                                                                  E_SYMBOL=E_SYMBOL, GO_SYMBOL=GO_SYMBOL)

            alignments.append(alignment)
            targets.append()

        print 'Alignment time: ' + str(time.time() - init_time)
        print 'Alignment: \n' + str(alignments)


        # Set vars
        teacher_forcing = []
        lengths = []
        max_lengths = [0] * len(alignments[0])

        # First calculate (max) lengths for all sequences
        for batch_index in range(batch_size):
            alignment = alignments[batch_index]

            # Calc temp true & max lengths for each transducer block
            lengths_temp = []
            alignment.insert(0, 0)  # This is so that the length calculation is done correctly
            for i in range(1, len(alignment)):
                lengths_temp.append(alignment[i] - alignment[i - 1] + 1)
                max_lengths[i - 1] = max(max_lengths[i - 1],
                                         lengths_temp[i - 1])  # For later use; how long each block is
            del alignment[0]  # Remove alignment index that we added
            lengths.append(lengths_temp)

        # Next modify so that each sequence is of equal length in each transducer block & targets have alignments
        for batch_index in range(batch_size):
            alignment = alignments[batch_index]

            # Modify targets so that it has the appropriate alignment
            offset = 0
            for e in alignment:
                targets[batch_index].insert(e + offset, E_SYMBOL)
                offset += 1

            # Modify so that all targets have same lengths in each transducer using PAD
            offset = 0
            for i in range(len(alignment)):
                for app in range(max_lengths[i] - lengths[batch_index][i]):
                    targets[batch_index].insert(offset + lengths[batch_index][i], PAD_SYMBOL)
                offset += max_lengths[i]

            # Modify targets for teacher forcing
            teacher_forcing_temp = list(targets[batch_index])
            teacher_forcing_temp.insert(0, GO_SYMBOL)
            teacher_forcing_temp.pop(len(teacher_forcing_temp) - 1)
            for i in range(len(teacher_forcing_temp)):
                if teacher_forcing_temp[i] == E_SYMBOL \
                        and targets[batch_index][i] != PAD_SYMBOL:
                    teacher_forcing_temp[i] = GO_SYMBOL

                if i + 1 < len(teacher_forcing_temp) and \
                        teacher_forcing_temp[i] == PAD_SYMBOL and \
                        teacher_forcing_temp[i + 1] != PAD_SYMBOL:
                    teacher_forcing_temp[i] = GO_SYMBOL

            teacher_forcing.append(teacher_forcing_temp)

        # Process targets back to time major
        targets = np.asarray(targets)
        targets = np.transpose(targets, axes=[1, 0])

        # See that teacher forcing are of correct format
        teacher_forcing = np.asarray(teacher_forcing)
        teacher_forcing = np.transpose(teacher_forcing, axes=[1, 0])

        return targets, teacher_forcing


