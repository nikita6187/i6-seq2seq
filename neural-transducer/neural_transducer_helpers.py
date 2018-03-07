import tensorflow as tf
import numpy as np
import copy
import sys
from multiprocessing import Process, Queue

# TODO: figure out a way to use the cons_manager (pickle?)
# TODO: inputs manager? maybe preprocess inputs into dic file which is then pickled and then loaded into the manager
# TODO: maybe not use queues? (probs ok for now)


class Aligner(object):

    def __init__(self, cons_manager):
        self.cons_manager = cons_manager
        # Init the interface for model loading
        self.max_blocks = self.inputs_full_raw = self.transducer_list_outputs = self.start_block = \
            self.encoder_hidden_init_fw = self.encoder_hidden_init_bw = \
            self.trans_hidden_init = self.teacher_forcing_targets = self.inference_mode = self.logits = \
            self.encoder_hidden_state_new_fw = self.encoder_hidden_state_new_bw = \
            self.transducer_hidden_state_new = None

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

        # Select first alignment if we have multiple with the same log prob (happens with ~1% probability in training)

        return current_alignments[0].alignment_locations

    def get_model(self, session, path):

        # Restore graph
        saver = tf.train.import_meta_graph(path + '.meta')
        print 'Here 1'
        try:
            saver.restore(session, path)
        except Exception as e:
            print str(e.message)
            sys.stdout.flush()
        print 'Here 2'
        # Setup constants
        graph = tf.get_default_graph()
        #self.end_symbol = graph.get_tensor_by_name(name='transducer_training/end_symbol:0')
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
        self.encoder_hidden_state_new_fw = graph.get_operation_by_name(name='transducer_training/encoder_hidden_state_new_fw').outputs[0]
        self.encoder_hidden_state_new_bw = graph.get_operation_by_name(name='transducer_training/encoder_hidden_state_new_bw').outputs[0]
        self.transducer_hidden_state_new = graph.get_operation_by_name(name='transducer_training/transducer_hidden_state_new').outputs[0]

    def run(self, queue_input, queue_output, init_path):
        temp_list = []  # Holds the processed data

        x = tf.get_variable('x', [1])
        x = tf.Print(x, [x], message='Hello')

        # Init session
        print 'Child: start'
        with tf.Session() as sess:

            print 'Run test: ' + str(sess.run([x]))
            sys.stdout.flush()

            # Do init graph loading
            self.get_model(sess, init_path)

            print 'Child: got model!'
            sys.stdout.flush()

            # Main loop
            while True:
                # Process new alignment
                if queue_input.empty() is False:
                    print 'Child: new inputs!'
                    sys.stdout.flush()
                    (inputs, target) = queue_input.get()  # Retrieve new data
                    new_alignment = self.get_alignment(sess, inputs=inputs, targets=target,
                                                       input_block_size=self.cons_manager.input_block_size,
                                                       transducer_max_width=self.cons_manager.transducer_max_width)
                    print 'Child: new alignments! ' + str(new_alignment)
                    sys.stdout.flush()
                    new_alignment.append((target, new_alignment))

                # Push all new alignments onto the output queue
                for a in temp_list:
                    if queue_output.full() is False:
                        queue_output.put(a)
                        temp_list.remove(a)


class AlignerManager(object):
    def __init__(self):
        self.alignment_dic = {}
        self.input_queue = Queue(10)
        self.output_queue = Queue(10)
        self.processes = []

    def start_aligners(self, amount_of_aligners, cons_manager, init_model_path):
        # Start new processes for aligners
        for i in range(amount_of_aligners):
            a = Aligner(cons_manager=cons_manager)
            p = Process(target=a.run, args=(self.input_queue, self.output_queue, init_model_path))
            p.daemon = True
            self.processes.append(p)
            p.start()

    def run_new_alignments(self, inputs, targets):
        # TODO: Rewrite the way inputs are managed
        batch_size = inputs.shape[1]
        for i in range(batch_size):
            if self.input_queue.full() is False:
                print 'Manager: Putting in new data: ' + str(targets)
                self.input_queue.put(obj=(np.reshape(inputs[:, i, :], newshape=(-1, 1, 1)), targets[i]))
                self.alignment_dic[str(targets[i])] = (None, np.reshape(inputs[:, i, :], newshape=(-1, 1, 1)))

    def retrieve_new_alignments(self):
        while self.output_queue.empty() is False:
            (target, alignment) = self.output_queue.get()
            print 'Manager: Got new alignments: ' + str(alignment)
            (_, inputs) = self.alignment_dic[str(target)]
            self.alignment_dic[str(target)] = (alignment, inputs)
        # TODO: save this in a file

# TODO: remanage entry input to this script (using args), which is then executed automatically from neural_transducer.py
