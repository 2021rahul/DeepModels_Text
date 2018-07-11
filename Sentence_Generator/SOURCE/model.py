# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:55:53 2018

@author: rahul.ghosh
"""

import tensorflow as tf
import config
import numpy as np
import neural_network
import utilities
import os


class MODEL():

    def __init__(self, CONFIG, training):
        self.config = CONFIG
        self.training = training
        self.inputs = tf.placeholder(shape=[self.config.batch_size, self.config.num_steps], dtype=tf.int32)
        self.labels = tf.placeholder(shape=[self.config.batch_size, self.config.num_steps], dtype=tf.int32)
        self.loss = None
        self.logits = None
        self.lr = tf.Variable(0.0, trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(self.lr, self.new_lr)
        self.initial_state = None
        self.final_state = None

    def build(self):
        embedding_layer = neural_network.Embedding_Layer([self.config.vocab_size, self.config.hidden_size])
        inputs = embedding_layer.lookup(self.inputs)

        if self.training and self.config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.config.keep_prob)

        rnn_graph = neural_network.RNN_Graph([self.config.hidden_size, self.config.num_layers], self.training, self.config.keep_prob, self.config.batch_size)
        self.initial_state = rnn_graph.initial_state
        output, state = rnn_graph.feed_forward(inputs, self.config)

        hidden_layer = neural_network.Softmax_Layer([self.config.hidden_size, self.config.vocab_size])
        self.logits = hidden_layer.feed_forward(output)
        self.logits = tf.reshape(self.logits, [self.config.batch_size, self.config.num_steps, self.config.vocab_size])
        loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.labels,
                                                tf.ones([self.config.batch_size, self.config.num_steps], dtype=tf.float32),
                                                average_across_timesteps=False,
                                                average_across_batch=True)
        self.loss = tf.reduce_sum(loss)
        self.final_state = state

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

    def train(self, data, model_name):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')
            state = session.run(self.initial_state)
            for epoch in range(self.config.max_max_epoch):
                avg_cost = 0
                lr_decay = self.config.lr_decay ** max(epoch+1-self.config.max_epoch, 0.0)
                self.assign_lr(session, self.config.learning_rate * lr_decay)
                total_batch = int((data.batch_len - 1)/self.config.num_steps)
                for batch in range(total_batch):
                    batch_X, batch_Y = data.generate_batch(self.config.batch_size, self.config.num_steps, batch)
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y, self.initial_state: state}
                    _, loss_val, state = session.run([optimizer, self.loss, self.final_state], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "Learning rate:", "{:.3f}".format(session.run(self.lr)))
            save_path = saver.save(session, model_name)
            print("Model saved in path: %s" % save_path)

    def test(self, data, model_name):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, model_name)
            total_batch = int((data.batch_len - 1)/self.config.num_steps)
            avg_cost = 0
            state = session.run(self.initial_state)
            for batch in range(total_batch):
                batch_X, batch_Y = data.generate_batch(self.config.batch_size, self.config.num_steps, batch)
                feed_dict = {self.inputs: batch_X, self.labels: batch_Y, self.initial_state: state}
                loss_val, state = session.run([self.loss, self.final_state], feed_dict=feed_dict)
                print("batch:", batch, " loss: ", loss_val)
                avg_cost += loss_val / total_batch
            print("cost =", "{:.3f}".format(avg_cost))

    def generate(self, data, model_name, num_sentences):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, model_name)
            words = data.reverse_dictionary
            state = session.run(self.initial_state)
            x = 2
            inputs = np.matrix([[x]])
            text = ""
            count = 0
            while count < num_sentences:
                feed_dict = {self.inputs: inputs, self.initial_state: state}
                logits, state = session.run([self.logits, self.final_state], feed_dict=feed_dict)
                output_probs = tf.nn.softmax(logits)
                x = utilities.sample(session.run(output_probs[0][0]), 0.9)
                if words[x] == "<eos>":
                    text += ".\n\n"
                    count += 1
                else:
                    text += " " + words[x]
                inputs = np.matrix([[x]])
        with open(os.path.join(config.OUT_DIR, "output.txt"), "w", encoding='utf8') as outfile:
            outfile.write(text)
