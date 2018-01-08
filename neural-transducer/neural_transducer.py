import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.layers import core as layers_core
import numpy as np
import copy

# Current bug in line 368 in BuildFullTransducer, tensors are sometimes not evaluated, and sometimes only evaluated
# when using the TF Print function (???), see: https://github.com/tensorflow/tensorflow/issues/15851

# NOTE: Time major

# Constants
input_dimensions = 1
vocab_size = 5
input_embedding_size = 20
encoder_hidden_units = 8
inputs_embedded = True
transducer_hidden_units = 8
batch_size = 1
GO_SYMBOL = vocab_size - 1  # TODO: Make these constants correct
END_SYMBOL = vocab_size
E_SYMBOL = vocab_size - 2
input_block_size = 3
log_prob_init_value = 0


# ---------------- Helper classes -----------------------

class Alignment(object):
    def __init__(self):
        self.alignment_position = (1, 1)  # x = position in target (y~), y = block index, both start at 1
        self.log_prob = log_prob_init_value  # The sum log prob of this alignment over the target indices
        self.alignment_locations = []  # At which indices in the target output we need to insert <e>
        self.last_state_transducer = np.zeros(shape=(2, 1, transducer_hidden_units))  # Transducer state

    def __compute_sum_probabilities(self, transducer_outputs, targets, transducer_amount_outputs):
        def get_prob_at_timestep(timestep):
            return transducer_outputs[timestep][0][targets[start_index + timestep]]  # TODO: Debug, uncomment below
            # return np.log(transducer_outputs[timestep][0][targets[start_index + timestep]])

        start_index = self.alignment_position[0] - transducer_amount_outputs  # The current position of this alignment
        prob = log_prob_init_value
        for i in range(0, transducer_amount_outputs):
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
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)


class Model(object):
    def __init__(self):
        self.encoder_inputs, self.encoder_inputs_length, self.encoder_hidden_state, \
        self.encoder_outputs, self.encoder_hidden_state_new = self.build_encoder_model()
        self.encoder_raw_outputs, self.trans_hidden_state, self.transducer_amount_outputs, \
        self.transducer_hidden_state_new, self.logits, self.decoder_prediction = self.build_transducer_model()

    def build_encoder_model(self):
        encoder_inputs = tf.Variable(tf.zeros(shape=(input_block_size, batch_size, input_dimensions)),
                                     dtype=tf.float32, name='encoder_inputs', trainable=False)
        encoder_inputs_length = tf.Variable([tf.shape(encoder_inputs)[0]], dtype=tf.int32,
                                            name='encoder_inputs_length', trainable=False)
        encoder_hidden_state = tf.Variable(tf.zeros(shape=(2, 1, encoder_hidden_units)), dtype=tf.float32,
                                           name='encoder_hidden_state')  # Save the state as one tensor

        if inputs_embedded is True:
            encoder_inputs_embedded = encoder_inputs
        else:
            encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

        # Build model
        encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

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

        return encoder_inputs, encoder_inputs_length, encoder_hidden_state, encoder_outputs, encoder_hidden_state_new

    def build_transducer_model(self):
        encoder_raw_outputs = tf.Variable(tf.zeros(shape=(input_block_size, 1, encoder_hidden_units)),
                                          dtype=tf.float32,
                                          name='encoder_raw_outputs')
        trans_hidden_state = tf.Variable(tf.zeros(shape=(2, 1, transducer_hidden_units)),
                                         dtype=tf.float32,
                                         name='trans_hidden_state')  # Save the state as one tensor
        transducer_amount_outputs = tf.Variable(0, dtype=tf.int32, name='transducer_amount_outputs',
                                                trainable=False)

        # Model building
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embeddings,
            start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
            end_token=END_SYMBOL)

        attention_states = tf.transpose(encoder_raw_outputs,
                                        [1, 0, 2])  # attention_states: [batch_size, max_time, num_units]

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            encoder_hidden_units, attention_states)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            tf.contrib.rnn.LSTMCell(transducer_hidden_units),
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

        return encoder_raw_outputs, trans_hidden_state, transducer_amount_outputs, transducer_hidden_state_new, \
               logits, decoder_prediction


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
    # inputs of shape: [max_time, 1, input_dimensions]
    # targets of shape: [time]

    # TODO: Manage new blocks                       [p, d]
    # TODO: Manage variables                        [p, d1]
    # TODO: block_index etc. to start at index 1    [p, d]

    def run_new_block(session, block_inputs, previous_alignments, block_index, transducer_max_width, targets,
                      total_blocks,
                      last_encoder_state):
        """
        Runs one block of the alignment process.
        :param session: The current TF session.
        :param block_inputs: The truncated inputs for the current block. Shape: [block_time, 1, input_dimensions]
        :param previous_alignments: List of alignment objects from previous block step.
        :param block_index: The index of the current new block.
        :param transducer_max_width: The max width of the transducer block.
        :param targets: The full target array of shape [time]
        :param total_blocks: The total amount of blocks.
        :param last_encoder_state: The encoder state of the previous step.
        :return: new_alignments as list of Alignment objects,
        last_encoder_state_new in shape of [2, 1, encoder_hidden_units]
        """

        # TODO: Final testing

        def run_encoder(session, encoder_state, inputs):
            """
            Runs the encoder on specified inputs. Returns the outputs and encoder state.
            :param session: Current session.
            :param encoder_state: Current Encoder state, shape [2, 1, encoder_hidden_units]
            :param inputs: The truncated inputs for this block. Shape [block_time, 1, input_dimensions]
            :return: Encoder ouputs [max_time, batch_size, encoder_hidden_units], Encoder state [2, 1, encoder_hidden_units]
            """

            # Inputs
            enc_out, enc_state = session.run([model.encoder_outputs, model.encoder_hidden_state_new],
                                             feed_dict={
                                                 model.encoder_inputs: inputs,
                                                 model.encoder_inputs_length: [inputs.shape[0]],
                                                 model.encoder_hidden_state: encoder_state,

                                             })
            # print 'Encoder output' + str(enc_out)
            """
            enc_out = np.random.uniform(-1.0, 1.0, size=(input_block_size, 1, encoder_hidden_units))
            enc_state = None
            """
            return enc_out, enc_state

        def run_transducer(session, transducer_state, encoder_outputs, transducer_width):
            """
            Runs a transducer on one block of inputs for transducer_amount_outputs.
            :param session: Current session.
            :param transducer_state: The last transducer state as [2, 1, transducer_hidden_units] tensor.
            :param encoder_outputs: The outputs of the encoder on the input. [block_time, 1, encoder_hidden_units]
            :param transducer_width: The amount of outputs the transducer should produce.
            :return: Transducer logits [transducer_width, batch_size(1), vocab_size],
            Transducer state [2, 1, transducer_hidden_units]
            """

            trans_out, trans_state = session.run([model.logits, model.transducer_hidden_state_new],
                                                 feed_dict={
                                                     model.encoder_raw_outputs: encoder_outputs,
                                                     model.trans_hidden_state: transducer_state,
                                                     model.transducer_amount_outputs: transducer_width
                                                 })
            # apply softmax on the outputs
            trans_out = softmax(trans_out, axis=2)

            """
            print ''
            print 'Is this right?: ' + str(np.transpose(trans_out, axes=[1, 0, 2]))
            print 'Transducer output: ' + str(trans_out)
            
            trans_out = np.asarray([[[0.1, 0.7 + np.random.uniform(-0.15, 0.15), 0.2]]] * transducer_width)
            trans_state = None
            """
            return trans_out, trans_state

        # Run encoder and get inputs
        block_encoder_outputs, last_encoder_state_new = run_encoder(session=session, encoder_state=last_encoder_state,
                                                                    inputs=block_inputs)

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
                trans_out, trans_state = run_transducer(session=session,
                                                        transducer_state=alignment.last_state_transducer,
                                                        encoder_outputs=block_encoder_outputs,
                                                        transducer_width=new_alignment_width + 1)  # +1 due to the last symbol being <e>
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

    for block in range(current_block_index, amount_of_input_blocks + 1):
        trunc_inputs = inputs[((block - 1) * input_block_size):block * input_block_size]

        current_alignments, last_encoder_state = run_new_block(session=session, block_inputs=trunc_inputs,
                                                               previous_alignments=current_alignments,
                                                               block_index=block,
                                                               transducer_max_width=transducer_max_width,
                                                               targets=targets, total_blocks=amount_of_input_blocks,
                                                               last_encoder_state=last_encoder_state)

    # Check if we've found an alignment, it should be one
    assert len(current_alignments) == 1

    # Offset by one due to how indexing was done internally has to be compensated
    return [x-1 for x in current_alignments[0].alignment_locations]


# ----------------- Training --------------------------

def build_full_transducer():
    # Inputs
    max_blocks = tf.placeholder(dtype=tf.int32, name='max_blocks')
    inputs_full_raw = tf.placeholder(shape=(None, batch_size, input_dimensions), dtype=tf.float32,
                                     name='inputs_full_raw')
    transducer_list_outputs = tf.placeholder(shape=(None,), dtype=tf.int32,
                                             name='transducer_list_outputs')  # amount to output per block

    # Turn inputs into tensor which is easily readable
    inputs_full = tf.reshape(inputs_full_raw, shape=[max_blocks, input_block_size, batch_size, input_dimensions])

    # Outputs
    outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_blocks)

    # Hidden states
    # TODO: make these correct
    encoder_hidden_init = tf.ones(shape=(2, 1, encoder_hidden_units))
    trans_hidden_init = tf.ones(shape=(2, 1, transducer_hidden_units))

    init_state = (0, outputs_ta, encoder_hidden_init, trans_hidden_init)

    def cond(current_block, outputs_int, encoder_hidden, trans_hidden):
        return current_block < max_blocks

    def body(current_block, outputs_int, encoder_hidden, trans_hidden):
        # Process encoder
        # TODO: check if encoder_inputs_length is correct
        model.encoder_inputs = model.encoder_inputs.assign(inputs_full[current_block])
        model.encoder_inputs_length = model.encoder_inputs_length.assign([tf.shape(model.encoder_inputs)[0]])
        model.encoder_hidden_state = model.encoder_hidden_state.assign(encoder_hidden)

        current_block = tf.Print(current_block, [model.encoder_inputs_length], message='Enc in lengths: ')

        # TODO: Error is SOMETIMES gone when using tf.Print
        current_block = tf.Print(current_block, [model.encoder_inputs], message='Enc in: ')

        # Flow data from encoder to transducer
        model.encoder_raw_outputs = model.encoder_raw_outputs.assign(model.encoder_outputs)
        model.trans_hidden_state = model.trans_hidden_state.assign(trans_hidden)
        model.transducer_amount_outputs = model.transducer_amount_outputs.assign(transducer_list_outputs[current_block])

        current_block = tf.Print(current_block, [model.encoder_raw_outputs], message='Trans in: ')

        # Note the outputs
        outputs_int = outputs_int.write(current_block, model.logits)

        return current_block + 1, outputs_int, model.encoder_hidden_state_new, model.transducer_hidden_state_new

    _, outputs_final, _, _ = tf.while_loop(cond, body, init_state)

    # Process outputs
    outputs = outputs_final.stack()  # Now the outputs are of shape [block, amount_of_trans_out, batch_size, vocab]
    outputs = tf.reshape(outputs, shape=(-1, 1, vocab_size))  # And now its [amount_outputs, batch_size, vocab]

    model.encoder_outputs = tf.Print(model.encoder_outputs, [model.encoder_outputs], message='Current block enc out: ')

    return max_blocks, inputs_full_raw, transducer_list_outputs, outputs, model.encoder_outputs


def build_training_step(transducer_outputs_logits):
    targets = tf.placeholder(shape=(None,), dtype=tf.int32, name='targets')
    targets_one_hot = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot,
                                                                     logits=transducer_outputs_logits)
    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return targets, train_op, loss


def apply_training_step(session, inputs, targets, input_block_size, transducer_max_width,
                        inp_max_blocks, inp_inputs_full_raw, inp_targets, inp_trans_list_out, out_train_op, out_loss):
    """
    Applies a training step to the transducer model. This method can be called multiple times from e.g. a loop.
    :param session: The current session.
    :param inputs: The full inputs. Shape: [max_time, 1, input_dimensions]
    :param targets: The full targets. Shape: [time]. Each entry is an index.
    :param input_block_size: The block width for the inputs.
    :param transducer_max_width: The max width for the transducer.
    :param inp_max_blocks: Ref to TF var that holds the amount of blocks in that the input will be processed.
    :param inp_inputs_full_raw: Ref to TF var that holds the full inputs.
    :param inp_targets: Ref to TF var that holds the targets.
    :param inp_trans_list_out: Ref to TF var that holds the list of how long the transducer should run for each block.
    :param out_train_op: Ref to TF var that holds the training operation.
    :param out_loss: Ref to TF var that holds the training loss info.
    :return:
    """

    # TODO: get_alignment and insert into targets                               [p]
    # TODO: calc length of each transducer block                                [p]
    # TODO: run_full transducer and apply training step                         [p]

    # Get alignment and insert it into the targets
    alignment = get_alignment(session=session, inputs=inputs, targets=targets, input_block_size=input_block_size,
                              transducer_max_width=transducer_max_width)
    for e in alignment:
        targets.insert(e, E_SYMBOL)

    print alignment
    # Calc length for each transducer block
    lengths = []
    alignment.insert(0, 0)  # This is so that the length calculation is done correctly
    for i in range(1, len(alignment)):
        lengths.append(alignment[i] - alignment[i-1])

    # Run training step
    _, loss = sess.run([out_train_op, out_loss], feed_dict={
        inp_max_blocks: len(lengths),
        inp_inputs_full_raw: inputs,
        inp_targets: targets,
        inp_trans_list_out: lengths
    })

    return loss


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


def test_get_alignment():
    testy_inputs = np.random.uniform(-1.0, 1.0, size=(10, 1, 5))
    print get_alignment(None, inputs=testy_inputs, targets=testy_targets, input_block_size=2, transducer_max_width=2)
    # TODO: test more vigorously


# ---------------------- Management -----------------------------

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Uncomment the following 5 lines to see error, we get an error in line 510/511 for some reason though
    """inp_max_blocks, inp_inputs_full_raw, inp_trans_list_out, out_outputs, enc_out = build_full_transducer()
    print sess.run([enc_out, out_outputs], feed_dict={
        inp_max_blocks: 3,
        inp_inputs_full_raw: np.ones(shape=(3 * input_block_size, 1, input_dimensions)),
        inp_trans_list_out: [1, 3, 2]
    })"""

    # Build training op
    # inp_max_blocks, inp_inputs_full_raw, inp_trans_list_out, out_outputs, enc_out = build_full_transducer()
    # targets, train_op, loss = build_training_step(out_outputs)

    # Apply training step
    """
    print apply_training_step(session=sess, inputs=np.ones(shape=(3 * input_block_size, 1, input_dimensions)),
                              input_block_size=input_block_size, targets=[1, 1, 1, 1], transducer_max_width=2,
                              inp_max_blocks=inp_max_blocks, inp_inputs_full_raw=inp_inputs_full_raw,
                              inp_trans_list_out=inp_trans_list_out, inp_targets=targets, out_train_op=train_op,
                              out_loss=loss)
    """

    print get_alignment(session=sess, inputs=np.ones(shape=(3 * input_block_size, 1, input_dimensions)),
                        input_block_size=input_block_size, targets=[1, 1, 1, 1], transducer_max_width=2)

