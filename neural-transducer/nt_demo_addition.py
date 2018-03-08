import os
from neural_transducer import ConstantsManager, Model, DataManager
import tensorflow as tf
import numpy as np
import dataset_loader

# Remaking the toy addition testing example from the paper

dir = os.path.dirname(os.path.realpath(__file__))
model_save = dir + '/addition/model_init'
input_save = dir + '/addition/inputs.npy'
target_save = dir + '/addition/targets.npy'
alignments_save = dir + '/addition/alignments'
cons_man_save = dir + '/addition/cons_manager'

vocab_ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SPACE']

transducer_width = 8

constants_manager = ConstantsManager(input_dimensions=1, input_embedding_size=11, inputs_embedded=False,
                                     encoder_hidden_units=100, transducer_hidden_units=200, vocab_ids=vocab_ids,
                                     input_block_size=1, beam_width=5, encoder_hidden_layers=1, transducer_max_width=8,
                                     path_to_model=model_save, path_to_inputs=input_save, path_to_targets=target_save,
                                     path_to_alignments=alignments_save, path_to_cons_manager=cons_man_save,
                                     amount_of_aligners=2)
model = Model(cons_manager=constants_manager)
init = tf.global_variables_initializer()

def lookup(i):
    return constants_manager.vocab_ids[i]


# -- Training ---

def get_feed_dic(batch_size):
    def get_random_numbers():
        a = np.random.randint(100, 499)     # That way any 2 sequences are always the same length
        b = np.random.randint(100, 499)
        c = a + b

        inputs = [int(d) for d in str(a)]
        inputs.append(10)  # Space
        inputs += list(reversed([int(d) for d in str(b)]))

        targets = list(reversed([int(d) for d in str(c)]))
        return inputs, targets

    inputs = []
    targets = []

    for i in range(batch_size):
        temp_inputs, temp_targets = get_random_numbers()
        inputs.append(temp_inputs)
        targets.append(temp_targets)

    inputs = np.asarray(inputs)
    inputs = np.transpose(inputs, axes=[1, 0])
    inputs = np.reshape(inputs, newshape=(-1, batch_size, 1))

    return inputs, targets


with tf.Session() as sess:
    sess.run(init)

    avg_loss = 0
    avg_over = 30

    # Generate dummy data
    inputs, targets = get_feed_dic(100)

    # Data manager
    data_manager = DataManager(constants_manager, full_inputs=inputs, full_targets=targets, model=model, session=sess)
    data_manager.run_new_alignments()

    # Apply training step
    for i in range(0, 40):

        # Apply training
        temp_loss = model.apply_training_step(session=sess, batch_size=4, data_manager=data_manager)
        avg_loss += temp_loss
        if i % avg_over == 0:
            avg_loss /= avg_over
            print 'Step: ' + str(i)
            print 'Loss: ' + str(avg_loss)
            avg_loss = 0


    # Save for inference later
    #model.save_model_for_inference(sess, model_save)


# -- Inference --
"""
with tf.Session() as sess2:

    inference = InferenceManager(session=sess2, beam_search=False, path=model_save,
                                 transducer_width=transducer_width, model=model, cons_manager=constants_manager)


    for x in range(10):
        i, t = get_feed_dic()

        print 'New inference test: '
        print np.reshape(i, shape=(-1)).tolist()
        print inference.run_inference(sess2, model_save, i, clean_e=False)[1]
        print str(map(lookup, t))
"""
