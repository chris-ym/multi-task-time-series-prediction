import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='RTNet', help='Model name: RTNet')
parser.add_argument('--num_data', type=int, default='10', help='number of balanced data')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_trans_enc', type=int, default=2, help='Number of transformer encoder [default: 2]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--loss_weight', type=int, default=1, help='Initial loss weight [default: 1]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
LOSS_WEIGHT = FLAGS.loss_weight
OPTIMIZER = FLAGS.optimizer
LOG_DIR = FLAGS.log_dir

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# next_time_pred pkgs
old_target=df[['target']]
df=df.drop(['target'],axis=1)


# Number of training samples.
sample_count = len(train_sample)
epochs = 50

# Number of warmup epochs.
warmup_epoch = 10

# Base learning rate after warmup.
learning_rate_base = 0.001

total_steps = int(epochs * sample_count / batch_size)

# Compute the number of warmup batches.
warmup_steps = int(warmup_epoch * sample_count / batch_size)

warm_up_lr = tf_util.WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)


# build training model and save
def train():
    with tf.Graph().as_default():
        i=0
        clf=create_model()
        clf.summary()
        history=clf.fit(np.asarray(train_cb),[np.asarray(train_reg),np.asarray(train_lbl)],epochs=max_epoch,batch_size=batch_size,callbacks=[callback], validation_data=(np.asarray(val_cb),[np.asarray(val_reg),np.asarray(val_lbl)])) 
        clf.save("ensemble_2SA_eval1_%s_combinemodel_v2_0to10_loss_weight_1of5.h5" % i)
        #for i in range(num_data)
        
