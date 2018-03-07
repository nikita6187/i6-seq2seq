import os
from neural_transducer import ConstantsManager, Model, InferenceManager
import tensorflow as tf
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
constants_manager = ConstantsManager(input_dimensions=20, input_embedding_size=20, inputs_embedded=True,
                                     encoder_hidden_units=8, transducer_hidden_units=8, vocab_ids=['0', '1', '2'],
                                     input_block_size=3, beam_width=5)
model = Model(cons_manager=constants_manager)
init = tf.global_variables_initializer()

# -- Training ---

with tf.Session() as sess:
    sess.run(init)

    # Apply training step
    for i in range(0, 2):
        print model.apply_training_step(session=sess,
                                        inputs=np.ones(shape=(5 * constants_manager.input_block_size,
                                                              1,
                                                              constants_manager.input_dimensions)),
                                        input_block_size=constants_manager.input_block_size, targets=[1, 2, 1, 2, 1, 2],
                                        transducer_max_width=2, training_steps_per_alignment=10)

    model.save_model_for_inference(sess, dir + '/model_save/model_test1')


# -- Inference --

with tf.Session() as sess2:
    inference = InferenceManager(session=sess2, beam_search=False, path=dir+'/model_save/model_test1',
                                 transducer_width=2, model=model, cons_manager=constants_manager)
    inference.run_inference(sess2, dir + '/model_save/model_test1',
                            full_inputs=np.ones(shape=(5 * constants_manager.input_block_size,
                                                       1,
                                                       constants_manager.input_dimensions)), clean_e=False)

