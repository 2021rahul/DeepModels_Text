# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:16:29 2018

@author: rahul.ghosh
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('..')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL')
SOURCE_DIR = os.path.join(ROOT_DIR, 'SOURCE')

# DATA FILES
TRAIN_FILENAME = "ptb.train.txt"
VALIDATION_FILENAME = "ptb.valid.txt"
TEST_FILENAME = "ptb.test.txt"

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# MODEL CONFIG
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class SmallGenConfig(object):
    """Small config. for generation"""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 1
    vocab_size = 10000
