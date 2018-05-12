import tensorflow as tf


class NeuralTransducerLayer(_ConcatInputLayer):
    """
    Performs a neural transducer based on the paper "A Neural Transducer": https://arxiv.org/abs/1511.04868.
    NOTE: Requires that the loss be neural_transducer_loss and be configured with the same parameters as this layer.
    """
    layer_class = "neural_transducer_layer"

    def __init__(self, transducer_hidden_units, num_outputs, transducer_max_width, input_block_size, go_symbol_index,
                 embedding_size, e_symbol_index, **kwargs):
        """
        Initialize the Neural Transducer.
        :param int transducer_hidden_units: Amount of units the transducer should have.
        :param int num_outputs: The size of the output layer, i.e. the size of the vocabulary including <E> and <GO>
        symbols.
        :param int transducer_max_width: The max amount of outputs in one NT block (including the final <E> symbol)
        :param int input_block_size: Amount of inputs to use for each NT block.
        :param int go_symbol_index: Index of go symbol that is used in the NT block. 0 <= go_symbol_index < num_outputs
        :param int embedding_size: Embeddding dimension size.
        :param int e_symbol_index: Index of e symbol that is used in the NT block. 0 <= e_symbol_index < num_outputs
        """

        super(NeuralTransducerLayer, self).__init__(**kwargs)

        # TODO: Debug everything
        # TODO: Optimize

        # Get embedding & go symbol
        embeddings = tf.Variable(tf.random_uniform([num_outputs, embedding_size], -1.0, 1.0), dtype=tf.float32,
                                 name='nt_embedding')

        # Ensure encoder is time major
        encoder_outputs = self.input_data.get_placeholder_as_time_major()

        # Do assertions
        assert 0 <= go_symbol_index <= num_outputs, 'NT: Go symbol outside possible outputs!'
        assert 0 <= e_symbol_index <= num_outputs, 'NT: E symbol outside possible outputs!'
        assert encoder_outputs.size_placeholder[0] % input_block_size == 0, 'NT: Input shape not corresponding to ' \
                                                                            'input block size (add padding or see if ' \
                                                                            'batch first).'

        # self.output.placeholder is of shape [transducer_max_width * amount_of_blocks, batch_size, num_outputs]
        self.output.placeholder = self.build_full_transducer(transducer_hidden_units=transducer_hidden_units,
                                                             embeddings=embeddings,
                                                             num_outputs=num_outputs,
                                                             input_block_size=input_block_size,
                                                             go_symbol_index=go_symbol_index,
                                                             transducer_max_width=transducer_max_width,
                                                             encoder_outputs=encoder_outputs)

        # TODO: Check if this is the correct format
        self.output.size_placeholder = tf.shape(self.output.placeholder)
        self.output.time_dim_axis = 0
        self.output.batch_dim_axis = 1

    def build_full_transducer(self, transducer_hidden_units, embeddings, num_outputs, input_block_size,
                              go_symbol_index, transducer_max_width, encoder_outputs):
        # - transducer_hidden_units (int32, static)
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
        # - encoder_outputs [max_time, batch_size, encoder_hidden]

        with tf.variable_scope('transducer_model'):

            # Get meta variables
            batch_size = tf.shape(encoder_outputs)[1]
            trans_hidden_init = tf.zeros([2, batch_size, transducer_hidden_units], dtype=tf.float32)
            max_blocks = tf.to_int32(tf.shape(encoder_outputs)[0]/input_block_size)
            transducer_list_outputs = tf.ones([max_blocks, batch_size]) * transducer_max_width
            inference_mode = 1.0
            teacher_forcing_targets = tf.ones([transducer_max_width * max_blocks, batch_size]) * go_symbol_index

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

                from tensorflow.python.layers import core as layers_core
                projection_layer = layers_core.Dense(num_outputs, use_bias=False)

                # Build previous state
                trans_hidden_c, trans_hidden_h = tf.split(trans_hidden_state, num_or_size_splits=2, axis=0)
                trans_hidden_c = tf.reshape(trans_hidden_c, shape=[-1, transducer_hidden_units])
                trans_hidden_h = tf.reshape(trans_hidden_h, shape=[-1, transducer_hidden_units])
                from tensorflow.contrib.rnn import LSTMStateTuple
                trans_hidden_state_t = LSTMStateTuple(trans_hidden_c, trans_hidden_h)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper,
                    decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=trans_hidden_state_t),
                    output_layer=projection_layer)
                outputs, transducer_hidden_state_new, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                            output_time_major=True,
                                                                                            maximum_iterations=transducer_max_output)
                logits = outputs.rnn_output  # logits of shape [max_time,batch_size,vocab_size]

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
    def get_out_data_from_opts(cls, num_outputs, transducer_max_width, input_block_size, **kwargs):
        " This is supposed to return a :class:`Data` instance as a template, given the arguments. "
        # TODO: make this correct
        data = get_concat_sources_data_template(sources)
        data = data.copy_as_time_major()
        batch_size = data.get_batch_dim()
        output_blocks = int(data.time_dimension()/input_block_size)
        data.shape = (transducer_max_width * output_blocks, batch_size, num_outputs)
        data.time_dim_axis = 0
        data.batch_dim_axis = 1
        # TODO: do we need to set size_placeholder?
        return data


class NeuralTransducerLoss(Loss):
    """
    The loss function that should be used with the NeuralTransducer layer. This loss function has the built in
    alignment algorithm from the original paper.
    """
    class_name = "neural_transducer_loss"

    class Alignment(object):
        """
        Class to manage the alignment generation in the NT.
        """

        def __init__(self, transducer_hidden_units, E_SYMBOL):
            """
            Alignment initiation.
            :param transducer_hidden_units: Amount of hidden units that the transducer should have.
            :param E_SYMBOL: The index of the <e> symbol.
            """
            import numpy as np
            self.alignment_position = (0, 1)  # first entry is position in target (y~), second is the block index
            self.log_prob = 0  # The sum log prob of this alignment over the target indices
            self.alignment_locations = []  # At which indices in the target output we need to insert <e>
            self.last_state_transducer = np.zeros(
                shape=(2, 1, transducer_hidden_units))  # Transducer state
            self.E_SYMBOL = E_SYMBOL  # Index of

        def __compute_sum_probabilities(self, transducer_outputs, targets, transducer_amount_outputs):
            """
            Computes the sum log probabilities of the outputs based on the targets.
            :param transducer_outputs: Softmaxed transducer outputs of one block.
            Size: [transducer_amount_outputs, 1, num_outputs]
            :param targets: List of targets.
            :param transducer_amount_outputs: The width of this transducer block.
            :return: The summed log prob for this block.
            """
            import numpy as np

            def get_prob_at_timestep(timestep):
                if timestep + start_index < len(targets):
                    # For normal operations
                    if transducer_outputs[timestep][0][targets[start_index + timestep]] <= 0:
                        return -10000000.0  # Some large negative number
                    else:
                        return np.log(transducer_outputs[timestep][0][targets[start_index + timestep]])
                else:
                    # For last timestep, so the <e> symbol
                    if transducer_outputs[timestep][0][self.E_SYMBOL] <= 0:
                        return -10000000.0  # Some large negative number
                    else:
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
            """
            self.alignment_locations.append(index)
            self.alignment_position = (index, block_index)
            self.log_prob += self.__compute_sum_probabilities(transducer_outputs, targets, transducer_amount_outputs)
            self.last_state_transducer = new_transducer_state

    @classmethod
    def softmax(cls, x, axis=None):
        import numpy as np
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def __init__(self, transducer_hidden_units, num_outputs, transducer_max_width, input_block_size, go_symbol_index,
                 e_symbol_index, **kwargs):
        """
        Initialize the Neural Transducer loss.
        :param int transducer_hidden_units: Amount of units the transducer should have.
        :param int num_outputs: The size of the output layer, i.e. the size of the vocabulary including <E> and <GO>
        symbols.
        :param int transducer_max_width: The max amount of outputs in one NT block (including the final <E> symbol)
        :param int input_block_size: Amount of inputs to use for each NT block.
        :param int go_symbol_index: Index of go symbol that is used in the NT block. 0 <= go_symbol_index < num_outputs
        :param int e_symbol_index: Index of e symbol that is used in the NT block. 0 <= e_symbol_index < num_outputs
        """
        super(NeuralTransducerLoss, self).__init__(**kwargs)
        self.transducer_hidden_units = transducer_hidden_units
        self.num_outputs = num_outputs
        self.transducer_max_width = transducer_max_width
        self.input_block_size = input_block_size
        self.go_symbol_index = go_symbol_index
        self.e_symbol_index = e_symbol_index

    def get_value(self):
        logits = self.output
        targets = self.target

        new_targets, mask = tf.py_func(func=self.get_alignment_from_logits_manager, inp=[logits, targets],
                                       Tout=(tf.int64, tf.bool), stateful=False)

        # Apply padding (convergence?), get loss and apply gradient
        # padding = tf.ones_like(new_targets) * self.cons_manager.PAD
        # new_targets = tf.where(mask, new_targets, padding)
        stepwise_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_targets, logits=logits)

        # Debugging
        stepwise_cross_entropy = tf.Print(stepwise_cross_entropy, [new_targets], message='Targets: ', summarize=100)
        stepwise_cross_entropy = tf.Print(stepwise_cross_entropy, [tf.argmax(logits, axis=2)], message='Argmax: ',
                                          summarize=100)
        # stepwise_cross_entropy = tf.Print(stepwise_cross_entropy, [stepwise_cross_entropy], message='CE PRE: ',
        #                                  summarize=1000)

        # Apply masking step AFTER cross entropy:
        zeros = tf.zeros_like(stepwise_cross_entropy)
        stepwise_cross_entropy = tf.where(mask, stepwise_cross_entropy, zeros)

        # Normalize CE based on amount of True (relevant) elements in the mask
        loss = tf.reduce_sum(stepwise_cross_entropy) / tf.to_float(tf.reduce_sum(tf.cast(mask, tf.float32)))
        return loss

    def get_alignment_from_logits(self, logits, targets, amount_of_blocks, transducer_max_width):
        """
        Finds the alignment of the target sequence to the actual output.
        :param logits: Logits from transducer, of size [transducer_max_width * amount_of_blocks, 1, vocab_size]
        :param targets: The target sequence of shape [time] where each entry is an index.
        :param amount_of_blocks: Amount of blocks in Neural Transducer.
        :param transducer_max_width: The max width of one transducer block.
        :return: Returns a list of indices where <e>'s need to be inserted into the target sequence, shape: [max_time, 1]
        (see paper) and a boolean mask for use with a loss function of shape [max_time, 1].
        """
        import numpy as np
        import copy
        # Split logits into list of arrays with each array being one block
        # of shape [transducer_max_width, 1, vocab_size]
        logits = np.reshape(logits, newshape=[logits.shape[0], 1, logits.shape[1]])

        split_logits = np.split(logits, amount_of_blocks)

        # print 'Raw logits: ' + str(softmax(split_logits[0][0:transducer_max_width], axis=2))

        def run_new_block(previous_alignments, block_index, transducer_max_width, targets,
                          total_blocks):
            """
            Runs one block of the alignment process.
            :param previous_alignments: List of alignment objects from previous block step.
            :param block_index: The index of the current new block.
            :param transducer_max_width: The max width of the transducer block.
            :param targets: The full target array of shape [time]
            :param total_blocks: The total amount of blocks.
            :return: new_alignments as list of Alignment objects
            """

            def run_transducer(current_block, transducer_width):
                # apply softmax on the correct outputs
                transducer_out = self.softmax(split_logits[current_block][0:transducer_width], axis=2)
                return transducer_out

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
                # TODO: Set this to correct log manager
                print '----------- New Alignment ------------'
                print 'Min index: ' + str(min_index)
                print 'Max index: ' + str(max_index)

                # new_alignment_index's value is equal to the index of y~ for that computation
                for new_alignment_index in range(min_index, max_index + 1):  # 1 so that the max_index is also used
                    print '---- New Index ----'
                    print 'Alignment index: ' + str(new_alignment_index)
                    # Create new alignment
                    new_alignment = copy.deepcopy(alignment)
                    print 'Alignment positions: ' + str(new_alignment.alignment_position)
                    new_alignment_width = new_alignment_index - new_alignment.alignment_position[0]
                    print 'New alignment width: ' + str(new_alignment_width)
                    trans_out = run_transducer(transducer_width=new_alignment_width + 1, current_block=block_index - 1)

                    new_alignment.insert_alignment(new_alignment_index, block_index, trans_out, targets,
                                                   new_alignment_width, None)
                    new_alignments.append(new_alignment)

            # Delete all overlapping alignments, keeping the highest log prob
            for a in reversed(new_alignments):
                for o in new_alignments:
                    if o is not a and a.alignment_position == o.alignment_position and o.log_prob > a.log_prob:
                        if a in new_alignments:
                            new_alignments.remove(a)

            return new_alignments

        # Manage variables
        current_block_index = 1
        current_alignments = [self.Alignment(transducer_hidden_units=self.transducer_hidden_units,
                                        E_SYMBOL=self.e_symbol_index)]

        # Do assertions to check whether everything was correctly set up.
        assert transducer_max_width * amount_of_blocks >= len(
            targets), 'transducer_max_width to small for targets'

        for block in range(current_block_index, amount_of_blocks + 1):
            # Run all blocks
            current_alignments = run_new_block(previous_alignments=current_alignments,
                                               block_index=block,
                                               transducer_max_width=transducer_max_width - 1,  # -1 due to offset for e
                                               targets=targets, total_blocks=amount_of_blocks)
            # for alignment in current_alignments:
            # print str(alignment.alignment_locations) + ' ' + str(alignment.log_prob)

        # Select first alignment if we have multiple with the same log prob (happens with ~1% probability in training)

        print 'Alignment:' + str(current_alignments[0].alignment_locations)

        def modify_targets(targets, alignment):
            # Calc lengths for each transducer block
            lengths_temp = []
            alignment.insert(0, 0)  # This is so that the length calculation is done correctly
            for i in range(1, len(alignment)):
                lengths_temp.append(alignment[i] - alignment[i - 1] + 1)
            del alignment[0]  # Remove alignment index that we added
            lengths = lengths_temp

            # Modify targets so that it has the appropriate alignment
            offset = 0
            for e in alignment:
                targets.insert(e + offset, self.cons_manager.E_SYMBOL)
                offset += 1

            # Modify so that all targets have same lengths in each transducer block using 0 (will be masked away)
            offset = 0
            for i in range(len(alignment)):
                for app in range(transducer_max_width - lengths[i]):
                    targets.insert(offset + lengths[i], 0)
                offset += transducer_max_width

            # Process targets back to time major
            targets = np.asarray([targets])
            targets = np.transpose(targets, axes=[1, 0])

            return targets, lengths

        m_targets, lengths = modify_targets(targets.tolist(), current_alignments[0].alignment_locations)
        # m_targets now of shape: [max_time, 1 (batch_size)] = [transducer_max_width * number_of_blocks, 1]

        # Create boolean mask for TF so that unnecessary logits are not used for the loss function
        # Of shape [max_time, batch_size], True where gradient data is kept, False where not

        def create_mask(lengths):
            mask = np.full(m_targets.shape, False)
            for i in range(amount_of_blocks):
                for j in range(lengths[i]):
                    mask[i*transducer_max_width:i*transducer_max_width + j + 1, 0] = True
            return mask

        mask = create_mask(lengths)

        # print 'Modified targets: ' + str(m_targets.T)
        # print 'Mask: ' + str(mask.T)

        return m_targets, mask

    def get_alignment_from_logits_manager(self, logits, targets):
        """
        Get the modified targets & mask.
        :param logits: Logits of shape [max_time, batch_size, vocab_size]
        :param targets: Targets of shape [max_time, batch_size]. Each entry denotes the index of the correct target.
        :return: modified targets of shape [max_time, batch_size, vocab_size]
        & mask of shape [max_time, batch_size]
        """
        import numpy as np
        logits = np.copy(logits)
        targets = np.copy(targets)

        # print 'Manager: Logits init shape: ' + str(logits.shape)

        m_targets = []
        masks = []

        amount_of_blocks = logits.shape[0]/self.transducer_max_width

        # Go over every sequence in batch
        for batch_index in range(logits.shape[1]):
            temp_target, temp_mask = self.get_alignment_from_logits(logits=logits[:, batch_index, :],
                                                                    targets=targets[:, batch_index],
                                                                    amount_of_blocks=amount_of_blocks,
                                                                    transducer_max_width=self.transducer_max_width)
            m_targets.append(temp_target)
            masks.append(temp_mask)

        # Concatenate the targets & masks on the time axis; due to padding m_targets are all the same
        m_targets = np.concatenate(m_targets, axis=1)
        masks = np.concatenate(masks, axis=1)

        return m_targets, masks
