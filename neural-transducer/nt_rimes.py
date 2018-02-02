import os
from neural_transducer import ConstantsManager, Model, InferenceManager
import tensorflow as tf
import numpy as np
import dataset_loader

# To make this work, put the RIMES 'train.0010' file into this directory

dir = os.path.dirname(os.path.realpath(__file__))
model_save = dir + '/rimes_save/rimes_save_1'
i, i_l, t, t_l = dataset_loader.load_from_file('train.0010')
bm = dataset_loader.BatchManager(i, i_l, t, t_l, pad='PAD')
print bm.lookup

transducer_width = 5

constants_manager = ConstantsManager(input_dimensions=i.shape[2], input_embedding_size=i.shape[2], inputs_embedded=True,
                                     encoder_hidden_units=256, transducer_hidden_units=256, vocab_ids=bm.lookup,
                                     input_block_size=67, beam_width=5)
model = Model(cons_manager=constants_manager)
init = tf.global_variables_initializer()


# -- Training ---

def get_feed_dic():
    inputs, _, targets, _ = bm.next_batch(batch_size=1)
    inputs = np.transpose(inputs, axes=[1, 0, 2])
    return inputs, targets.tolist()

with tf.Session() as sess:
    sess.run(init)

    avg_loss = 0
    avg_over = 10

    # Apply training step
    for i in range(0, 200):

        new_in, new_targets = get_feed_dic()

        temp_loss = model.apply_training_step(session=sess, inputs=new_in, input_block_size=constants_manager.input_block_size,
                                        targets=new_targets, transducer_max_width=transducer_width,
                                        training_steps_per_alignment=10)
        avg_loss += temp_loss
        if i % avg_over == 0:
            avg_loss /= avg_over
            print 'Loss: ' + str(avg_loss)
            avg_loss = 0


    # Save for inference later
    model.save_model_for_inference(sess, dir + '/rimes_save/model_test1')