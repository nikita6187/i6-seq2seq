import os
from neural_transducer import ConstantsManager, Model, DataManager
import tensorflow as tf
import numpy as np
import dataset_loader
import sys
import time
import datetime
import matplotlib.pyplot as plt

# USAGE:
# Param 1: Device (e.g. CPU:0)
# Param 2: Amount of aligners (e.g. 5)
# Param 3: Debug device (True/False)
# Param 4: Max cores to use for TF (e.g. 5)
# Param 5: Run offline alignments (True) or not (False)
# Param 6: Path & Prefix of initial model load (e.g. ../model_800)
# Param 7: Load in pre-computed alignments (True/False)
# Param 8: Use greedy for online alignments (True/False)

# To make this work, put the RIMES 'train.0010' file into this directory


def main():

    init_time_str = str(datetime.datetime.now())
    dir = os.path.dirname(os.path.realpath(__file__))
    i = []
    i_l = []
    t = []
    t_l = []

    # We remove very long sequences (over 300 in length)
    for iteration in range(1, 2):  # TODO: 11
        print '/rimes/training-data/train.00{0:02d}'.format(iteration)
        temp_i, temp_i_l, temp_t, temp_t_l = dataset_loader.load_from_file(
            dir + '/rimes/training-data/train.00{0:02d}'.format(iteration),
            max_length_input=502,
            max_length_target=18)
        # Remove all sequences above 300 length

        to_remove = np.argwhere(temp_i_l >= 300)
        if len(to_remove) > 0:
            to_remove = to_remove[0]
            print 'Removing: ' + str(to_remove)
            temp_i = np.delete(temp_i, to_remove, axis=0)
            temp_i_l = np.delete(temp_i_l, to_remove, axis=0)
            temp_t = np.delete(temp_t, to_remove, axis=0)
            temp_t_l = np.delete(temp_t_l, to_remove, axis=0)

        i.append(temp_i)
        i_l.append(temp_i_l)
        t.append(temp_t)
        t_l.append(temp_t_l)
        """
        print '\nShapes:'
        print temp_i.shape
        print temp_i_l.shape
        print temp_t.shape
        print temp_t_l.shape
        """

    i = np.concatenate(i, axis=0)
    i_l = np.concatenate(i_l, axis=0)
    t = np.concatenate(t, axis=0)
    t_l = np.concatenate(t_l, axis=0)

    # Cut down to correct size
    i = i[:, 0:300, :]

    # Get size:
    print 'Size of inputs: ' + str(sys.getsizeof(i))
    print 'Shape of inputs: ' + str(i.shape)
    print 'Total amount of sequences: ' + str(len(i_l))

    # Assertions that everything is ok
    assert i.shape[0] == i_l.shape[0] == t.shape[0] == t_l.shape[0], 'Incorrect sequence amounts!'

    # Vocab processing and shit
    bm = dataset_loader.BatchManager(i, i_l, t, t_l, pad='PAD')
    print bm.lookup

    model_save = dir + '/rimes/model_init'
    input_save = dir + '/rimes/inputs.npy'
    target_save = dir + '/rimes/targets.npy'
    alignments_save = dir + '/rimes/alignments'
    cons_man_save = dir + '/rimes/cons_manager'

    constants_manager = ConstantsManager(input_dimensions=i.shape[2], input_embedding_size=i.shape[2], inputs_embedded=True,
                                         encoder_hidden_units=256, transducer_hidden_units=512, vocab_ids=bm.lookup,
                                         input_block_size=25, beam_width=5, encoder_hidden_layers=1, transducer_max_width=8,
                                         path_to_model=model_save, path_to_inputs=input_save, path_to_targets=target_save,
                                         path_to_alignments=alignments_save, path_to_cons_manager=cons_man_save,
                                         amount_of_aligners=int(sys.argv[2]), device_to_run=str(sys.argv[1]),
                                         device_soft_placement=True, debug_devices=((sys.argv[3]).lower() == 'true'),
                                         max_cores=int(sys.argv[4]))

    with tf.device(constants_manager.device_to_run):  # Set device here
        model = Model(cons_manager=constants_manager)

    init = tf.global_variables_initializer()

    # -- Training ---

    # TODO: Dev/Test split and inference testing

    def get_feed_dic():
        inputs, _, targets, _ = bm.next_batch(batch_size=1)
        inputs = np.transpose(inputs, axes=[1, 0, 2])
        return inputs, targets.tolist()

    def convert_target_to_int(targets_list):
        for target_seq in targets_list:
            for i in range(len(target_seq)):
                target_seq[i] = bm.lookup_letter(target_seq[i])

        return targets_list

    config = tf.ConfigProto(allow_soft_placement=constants_manager.device_soft_placement,
                            log_device_placement=constants_manager.debug_devices,
                            device_count={'CPU': constants_manager.max_cores},
                            inter_op_parallelism_threads=constants_manager.max_cores,
                            intra_op_parallelism_threads=constants_manager.max_cores)
    config.gpu_options.allow_growth = True

    run_offline_alignments = sys.argv[5].lower() == 'true'

    with tf.Session(config=config) as sess:
        sess.run(init)

        # Try loading in prev built model from param 6 if it exists
        if len(sys.argv) >= 7 and os.path.isfile(sys.argv[6] + '.meta') is True:
            model.load_model(sess, sys.argv[6])

        # For benchmarking
        inputs = np.transpose(i, axes=[1, 0, 2])  # Time major
        targets = t.tolist()  # We need batch major lists for targets
        targets = convert_target_to_int(targets)

        init_time = time.time()

        use_greedy = sys.argv[8].lower() == 'true'
        print 'Using greedy: ' + str(use_greedy)
        data_manager = DataManager(constants_manager, full_inputs=inputs, full_targets=targets, model=model,
                                   session=sess, online_alignments=False, use_greedy=use_greedy)

        if run_offline_alignments is True:
            if sys.argv[7].lower() == 'true':
                data_manager.load_in_alignments()
            else:
                data_manager.run_new_alignments()
            print 'Time Needed for Alignments: ' + str(time.time() - init_time)
        else:
            data_manager.set_online_alignment(True)

        # Run training
        # TODO: make iteration count correct
        t__1 = time.time()

        for i in range(22):
            loss = model.apply_training_step(session=sess, batch_size=1, data_manager=data_manager)
            t_0 = time.time() - t__1
            t__1 = time.time()
            print 'Loss: ' + str(loss)

            with open(dir + '/rimes/output_' + init_time_str + '.txt', 'a') as myfile:
                myfile.write('\nLoss: ' + str(loss))
                myfile.write('\nTime: ' + str(t_0))

            # Switch to offline alignments after 1000 batches & run new alignmentsinput_block_size
            if i == 1000 and run_offline_alignments is False and use_greedy is False:
                data_manager.set_online_alignment(False)
                data_manager.run_new_alignments()

            # Save the model every 20 iterations
            if i % 20 == 0:
                model.save_model_for_inference(session=sess, path_name=dir + '/checkpoint/test_rimes_' + str(i))

        # Display correlation
        print constants_manager.alc_correlation_data
        x_data = [x[0] for x in constants_manager.alc_correlation_data]
        y_data = [x[1] for x in constants_manager.alc_correlation_data]
        plt.plot(x_data, y_data, 'go')
        plt.show()

        print 'Total Time Needed: ' + str(time.time() - init_time)

# TODO: load inference manager and use validation sets to test out


if __name__ == '__main__':
    main()
