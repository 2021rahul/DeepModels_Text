# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf


class Embedding_Layer():

        def __init__(self, shape):
            self.embedding = tf.get_variable("embedding", shape=shape, dtype=tf.float32)

        def lookup(self, input_data):
            output = tf.nn.embedding_lookup(self.embedding, input_data)
            return output


class Softmax_Layer():

    def __init__(self, shape):
        self.weights = tf.get_variable("softmax_w", shape=shape, dtype=tf.float32)
        self.biases = tf.get_variable("softmax_b", shape=[shape[1]], dtype=tf.float32)

    def feed_forward(self, input_data):
        logits = tf.nn.xw_plus_b(input_data, self.weights, self.biases)
        return logits


class RNN_Graph():

    def __init__(self, shape, training, keep_prob, batch_size):

        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(shape[0],
                                                forget_bias=0.0,
                                                state_is_tuple=True,
                                                reuse=not training)
            if training and keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
            return cell

        self.model = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(shape[1])], state_is_tuple=True)
        self.initial_state = self.model.zero_state(batch_size, dtype=tf.float32)

    def feed_forward(self, input_data, config):
        input_data = tf.unstack(input_data, num=config.num_steps, axis=1)
        output, state = tf.nn.static_rnn(self.model, input_data, initial_state=self.initial_state)
        output = tf.reshape(tf.concat(output, 1), [-1, config.hidden_size])
        return output, state
