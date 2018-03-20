#!/usr/bin/env python2
import tensorflow as tf
import numpy as np
import copy
import sys
from multiprocessing import Process, Queue
import cPickle
from neural_transducer import ConstantsManager, Alignment
import time
import psutil
from pympler import asizeof
import subprocess


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class AlignerWorker(object):

    def __init__(self, cons_manager, cpu_core):
        self.cons_manager = cons_manager
        # Init the interface for model loading
        self.max_blocks = self.inputs_full_raw = self.transducer_list_outputs = self.start_block = \
            self.encoder_hidden_init_fw = self.encoder_hidden_init_bw = \
            self.trans_hidden_init = self.teacher_forcing_targets = self.inference_mode = self.logits = \
            self.encoder_hidden_state_new_fw = self.encoder_hidden_state_new_bw = \
            self.transducer_hidden_state_new = None
        self.full_time_needed_transducer = 0
        self.cpu_core = cpu_core

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
            sys.stdout.flush()

        # Select first alignment if we have multiple with the same log prob (happens with ~1% probability in training)

        print 'Full time needed for transducer: ' + str(self.full_time_needed_transducer)

        return current_alignments[0].alignment_locations

    def get_model(self, path):
        # Restore graph
        saver = tf.train.import_meta_graph(path + '.meta')

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
        return saver

    def run(self, queue_input, queue_output, init_path):
        temp_list = []  # Holds the processed data

        # Init session
        config = tf.ConfigProto(allow_soft_placement=self.cons_manager.device_soft_placement,
                                log_device_placement=self.cons_manager.debug_devices,
                                device_count={'CPU': self.cons_manager.max_cores},
                                inter_op_parallelism_threads=self.cons_manager.max_cores,
                                intra_op_parallelism_threads=self.cons_manager.max_cores)
        config.gpu_options.allow_growth = True

        print 'Child process alive: ' + str(self.cpu_core)
        sys.stdout.flush()

        # Do init graph loading
        with tf.device('/cpu:' + str(self.cpu_core)):
            saver = self.get_model(init_path)

        with tf.Session(config=config) as sess:
            sys.stdout.flush()

            # Restore prev session
            saver.restore(sess, init_path)

            sys.stdout.flush()

            # Main loop
            while queue_input.empty() is False:
                # Process new alignment
                if queue_input.empty() is False:
                    (inputs, target) = queue_input.get()  # Retrieve new data
                    if inputs is not None and target is not None:
                        init_time = time.time()
                        #new_alignment = self.get_alignment(sess, inputs=inputs, targets=target,
                        #                                   input_block_size=self.cons_manager.input_block_size,
                        #                                   transducer_max_width=self.cons_manager.transducer_max_width)
                        new_alignment = [0, 1, 2, 3]
                        temp_list.append((inputs.tostring(), new_alignment))
                        print 'Aligner time needed full: ' + str(time.time() - init_time)
                        sys.stdout.flush()

                # Debugging
                sys.stdout.flush()

                # Push all new alignments onto the output queue
                for a in temp_list:
                    if queue_output.full() is False:
                        queue_output.put(a)
                        temp_list.remove(a)
                time.sleep(1)

            print 'Child process dead.'
            sys.stdout.flush()


class AlignerManager(object):
    def __init__(self, cons_manager):
        self.alignment_dic = {}  # key = inputs.tostring, value = (alignment)
        self.input_queue = Queue(10 * cons_manager.amount_of_aligners)
        self.output_queue = Queue(10 * cons_manager.amount_of_aligners)
        self.processes = []
        self.cons_manager = cons_manager

    def start_aligners(self):
        # Start new processes for aligners
        for i in range(self.cons_manager.amount_of_aligners):
            a = AlignerWorker(cons_manager=self.cons_manager, cpu_core=i+1)
            p = Process(target=a.run, args=(self.input_queue, self.output_queue, self.cons_manager.path_to_model))
            p.daemon = True
            self.processes.append(p)
            p.start()

    def run_new_alignments(self, inputs, targets):
        batch_size = inputs.shape[1]
        i = 0
        init_time = time.time()
        temp_debug_time = time.time()

        # Debugging
        process_data = []
        for p in self.processes:
            process_data.append(psutil.Process(p.pid))

        # Run for the whole inputs
        while i < batch_size:
            # Send new data out
            if self.input_queue.full() is False:
                self.input_queue.put(obj=(
                        np.reshape(inputs[:, i, :], newshape=(-1, 1, self.cons_manager.input_dimensions)),
                        targets[i]))
                self.alignment_dic[inputs[:, i, :].tostring()] = None
                i += 1
            # Receive new data
            self.retrieve_new_alignments()

            # Monitoring
            if time.time() - temp_debug_time > 2:
                mem_usage = 0
                temp_debug_time = time.time()
                for p in process_data:
                    mem_usage += p.memory_info().rss
                for p in self.processes:
                    # TODO: test if this works on cluster
                    f = open('/proc/{pid}/stat'.format(pid=str(p.pid)), 'rb')
                    print '\n Process running on core: ' + str(f.read().split(' ')[-14])
                    f.close()

                mem_usage = float(mem_usage)/(1024 * 1024 * 1024) * 10
                sys.stdout.write(
                    '\n Progress: {0:02.3f}% / Time running: {1:08d} / Memory Usage: {2:.3f}G / Amount of child processes: {3:02d}'.format(
                        float(i) / batch_size * 100, int(time.time() - init_time), mem_usage, len(self.processes)))
                sys.stdout.flush()

        # Wait for cleanup:
        time.sleep(5)  # Give another 5 seconds to clean up
        while self.input_queue.empty() is False:
            self.retrieve_new_alignments()

        # Finally process results into new dictionary

        save_dic = {}  # Now: key = inputs.tostring, value = alignment
        for key in self.alignment_dic:
            # In case of errors
            if self.alignment_dic[key] is not None:
                save_dic[key] = self.alignment_dic[key]
        file_alignments = open(self.cons_manager.path_to_alignments, 'wb')
        cPickle.dump(save_dic, file_alignments)
        file_alignments.close()

    def retrieve_new_alignments(self):
        while self.output_queue.empty() is False:
            (inputs_hash, alignment) = self.output_queue.get()
            self.alignment_dic[inputs_hash] = alignment


def main():
    # Get cons manager
    path_to_cons_manager = sys.argv[1]
    cons_man_file = open(path_to_cons_manager, 'rb')
    cons_manager = cPickle.load(cons_man_file)
    cons_man_file.close()

    # Make alignment manager
    align_manager = AlignerManager(cons_manager)
    align_manager.start_aligners()

    # Load inputs and targets
    inputs = np.load(cons_manager.path_to_inputs)
    targets = np.load(cons_manager.path_to_targets).tolist()
    align_manager.run_new_alignments(inputs, targets)


if __name__ == '__main__':
    main()
