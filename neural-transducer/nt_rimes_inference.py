import os
from neural_transducer import ConstantsManager, Model, DataManager, InferenceManager
import tensorflow as tf
import numpy as np
import dataset_loader
import sys
import time
import datetime
import matplotlib.pyplot as plt

# USAGE:
# Param 1: Device (e.g. CPU:0)
# Param 2: Debug device (True/False)
# Param 3: Max cores to use for TF (e.g. 5)
# Param 4: Path & Prefix of initial model load (e.g. ../model_800)

# To make this work, put the RIMES 'train.0010' file into this directory

# TODO: Load in alphabet from training data! Or else order is incorrect; using batch manager


def get_correct_alphabet():
    dir = os.path.dirname(os.path.realpath(__file__))
    i = []
    i_l = []
    t = []
    t_l = []

    # We remove very long sequences (over 300 in length)
    for iteration in range(1, 11):  # TODO: 11
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

    i = np.concatenate(i, axis=0)
    i_l = np.concatenate(i_l, axis=0)
    t = np.concatenate(t, axis=0)
    t_l = np.concatenate(t_l, axis=0)

    # Cut down to correct size
    i = i[:, 0:300, :]

    # Vocab processing and shit
    bm = dataset_loader.BatchManager(i, i_l, t, t_l, pad='PAD')
    return bm


def main():

    dir = os.path.dirname(os.path.realpath(__file__))
    i = []
    i_l = []
    t = []
    t_l = []

    # We remove very long sequences (over 300 in length)
    for iteration in range(1, 11):  # TODO: 11
        print '/rimes/training-data/valid.00{0:02d}'.format(iteration)
        temp_i, temp_i_l, temp_t, temp_t_l = dataset_loader.load_from_file(
            dir + '/rimes/training-data/valid.00{0:02d}'.format(iteration),
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
    bm = get_correct_alphabet()
    print 'Lookup: ' + str(bm.lookup)

    model_save = dir + '/rimes/model_init'
    input_save = dir + '/rimes/inputs.npy'
    target_save = dir + '/rimes/targets.npy'
    alignments_save = dir + '/rimes/alignments'
    cons_man_save = dir + '/rimes/cons_manager'

    # Note: set input_block_size correctly
    # TODO: note input block size!
    constants_manager = ConstantsManager(input_dimensions=i.shape[2], input_embedding_size=i.shape[2], inputs_embedded=True,
                                         encoder_hidden_units=512, transducer_hidden_units=1024, vocab_ids=bm.lookup,
                                         input_block_size=100, beam_width=5, encoder_hidden_layers=3, transducer_max_width=8,
                                         path_to_model=model_save, path_to_inputs=input_save, path_to_targets=target_save,
                                         path_to_alignments=alignments_save, path_to_cons_manager=cons_man_save,
                                         amount_of_aligners=4, device_to_run=str(sys.argv[1]),
                                         device_soft_placement=True, debug_devices=((sys.argv[2]).lower() == 'true'),
                                         max_cores=int(sys.argv[3]))

    with tf.device(constants_manager.device_to_run):  # Set device here
        model = Model(cons_manager=constants_manager)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=constants_manager.device_soft_placement,
                            log_device_placement=constants_manager.debug_devices,
                            device_count={'CPU': constants_manager.max_cores},
                            inter_op_parallelism_threads=constants_manager.max_cores,
                            intra_op_parallelism_threads=constants_manager.max_cores)
    config.gpu_options.allow_growth = True

    def convert_target_to_int(targets_list):
        for target_seq in targets_list:
            for i in range(len(target_seq)):
                target_seq[i] = bm.lookup_letter(target_seq[i])
        return targets_list

    with tf.Session(config=config) as sess:
        sess.run(init)

        # Load in data
        inputs = np.transpose(i, axes=[1, 0, 2])  # Time major
        targets = t.tolist()  # We need batch major lists for targets
        targets = convert_target_to_int(targets)
        data_manager = DataManager(constants_manager, full_inputs=inputs, full_targets=targets, model=model,
                                   session=sess, online_alignments=False, use_greedy=True, inference=True)

        # Load in inference
        inference_manager = InferenceManager(cons_manager=constants_manager)

        # Rebuild graph
        inference_manager.build_greedy_inference(path=sys.argv[4],
                                                 session=sess)

        for _ in range(20):
            # Try out inference
            inp, targ, _ = data_manager.get_new_random_sample()

            def lookup(i):
                return constants_manager.vocab_ids[i]
            targ = map(lookup, targ)
            print '\n'
            print 'Inference Run: '
            print 'Inferred data:'
            print inference_manager.run_inference(session=sess, full_inputs=inp, clean_e=True)[1]
            print 'Ground truth: '
            print targ


if __name__ == '__main__':
    main()
