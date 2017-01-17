from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from build import PTBModel
import reader
import random
import os

class SmallGenConfig(object):
  """Small config. for generation"""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 1
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 10000
  
def sample(a, temperature=1.0):
  a = np.log(a) / temperature
  a = np.exp(a) / np.sum(np.exp(a))
  r = random.random() # range: [0,1)
  total = 0.0
  for i in range(len(a)):
    total += a[i]
    if total>r:
      return i
  return len(a)-1
  
def generate_sentences(train_path , model_path , num_sentences):
  gen_config = SmallGenConfig()
  
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-gen_config.init_scale,
                                                gen_config.init_scale)    
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=False, config=gen_config)

    # Restore variables from disk.
    saver = tf.train.Saver() 
    saver.restore(session, model_path)
    print("Model restored from file " + model_path)
    
    words = reader.get_vocab(train_path)
    
    state = m.initial_state.eval()
    x = 2 # the id for '<eos>' from the training set
    input = np.matrix([[x]])  # a 2D numpy matrix 
    
    text = ""
    count = 0
    while count < num_sentences:
      output_probs, state = session.run([m.output_probs, m.final_state],
                                   {m.input_data: input,
                                    m.initial_state: state})
      x = sample(output_probs[0], 0.9)
      if words[x]=="<eos>":
        text += ".\n\n"
        count += 1
      else:
        text += " " + words[x]
      # now feed this new word as input into the next iteration
      input = np.matrix([[x]]) 
      
  return text
  
cwd = os.getcwd()
train_path = os.path.join(cwd , "/simple-examples/data/ptb.train.txt")
model_path = os.path.join(cwd , "/MODELS/model.cpkt")
num_sentences = 10
print generate_sentences(train_path,model_path,num_sentences)
