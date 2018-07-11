# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:00:55 2018

@author: rahul.ghosh
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import data
import model
import config
import tensorflow as tf


if __name__ == "__main__":
    with tf.Graph().as_default():
        # LOAD CONFIG
        model_config = config.MediumConfig()
        eval_config = config.MediumConfig()
        gen_config = config.SmallGenConfig()
        eval_config.batch_size = 1
        eval_config.num_steps = 1
        # READ DATA
        train_data = data.PTB_DATA()
        train_data.load_data(config.TRAIN_FILENAME, model_config.batch_size)
#        valid_data = data.PTB_DATA()
#        valid_data.load_data(config.VALIDATION_FILENAME, model_config.batch_size)
        test_data = data.PTB_DATA()
        test_data.load_data(config.TEST_FILENAME, eval_config.batch_size)
        # BUILD MODEL
        initializer = tf.random_uniform_initializer(-model_config.init_scale,
                                                    model_config.init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model = model.MODEL(model_config, training=True)
                train_model.build()
#        with tf.name_scope("Validate"):
#            with tf.variable_scope("Model", reuse=True, initializer=initializer):
#                valid_model = model.MODEL(model_config, training=False)
#                valid_model.build()
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = model.MODEL(eval_config, training=False)
                test_model.build()
        with tf.name_scope("Generate"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                gen_model = model.MODEL(gen_config, training=False)
                gen_model.build()
        # TRAIN MODEL
        model_name = os.path.join(config.MODEL_DIR, "model" + str(model_config.batch_size) + "_" + str(model_config.max_max_epoch) + ".ckpt")
        #train_model.train(train_data, model_name)
        # TEST MODEL
        #test_model.test(test_data, model_name)
        # GENERATE MODEL
        num_sentences = 10
        gen_model.generate(train_data, model_name, num_sentences)
