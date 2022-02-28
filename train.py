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
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='RTNet', help='Model name: RTNet')
parser.add_argument('--mode', default='training_mode', help='traing mode[default: training_mode]')
parser.add_argument('--num_data', type=int, default='10', help='number of balanced data')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_trans_enc', type=int, default=2, help='Number of transformer encoder [default: 2]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]') 
parser.add_argument('--loss_weight', type=int, default=1, help='Initial loss weight [default: 100]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
## transformer encoder default setting
parser.add_argument('--d_k', type=int, default=64, help='k value of self attenion [default: 64]')
parser.add_argument('--d_v', type=int, default=64, help='v value of self attenion [default: 64]')
parser.add_argument('--ff_dim', type=int, default=256, help='ffdim [default: 256]')
parser.add_argument('--n_heads', type=int, default=2, help='n_heads of muitihead attention [default: 2]')
FLAGS = parser.parse_args()

d_k = FLAGS.d_k
d_v = FLAGS.d_v
ff_dim = FLAGS.ff_dim
n_heads = FLAGS.n_heads

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # backup of model definition
os.system('cp train.py %s' % (LOG_DIR)) # backup of train process
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#### read train & test data#############################
#### Import data (different methods)

def get_data(data_name):
    import os
    import pickle
    name, extension = os.path.splitext(path)
    if extension == ".csv":
        data=pd.read_csv("%s.csv"%data_name)
    elif extension == ".pkl":
        f=open("%s.pkl"%data_name,'rb')
        data=pickle.load(f)
        f.close()
    else:
        
    return data
#######################################################
    

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    
####load model weight

if FLAGS.model == "RTNet":
    model_weights = os.getcwd() + '\model_weights\best_RTNet_weights.h5'
elif FLAGS.model == "CTNet":
    model_weights = os.getcwd() + '\model_weights\best_cTNet_weights.h5'
    
####definition of model
def setup_model():
    if FLAGS.model == "RTNet":
        model = create_model(data, data2)
    elif FLAGS.model == "CTNet":
        model = create_model(data)
    else:
        print("None of the constructed model!")
        
    ##set loss weight
    model.compile(loss=['mse','binary_crossentropy'], optimizer=adam,metrics=['acc'],loss_weights=[ 1., FLAGS.loss_weight])
    
    
#### train function (version 2)####
def train():
    if FLAGS.mode == "eager_mode":
        000
        
    else:
        
        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        start_time = time.time()
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        end_time = time.time() - start_time
        print(f'Total Training Time: {end_time}')
            

#########################

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
