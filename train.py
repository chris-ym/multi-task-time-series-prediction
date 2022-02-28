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

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='RTNet', help='Model name: RTNet')
parser.add_argument('--pretrained', type=str_to_bool, default=False, help='boolean value')
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
    model_weights = os.getcwd() + '\model_weights\best_CTNet_weights.h5'
    
####definition of model
look_ahead_mask = transformer_encoder.create_look_ahead_mask(seq_len)

def setup_model():
    if FLAGS.model == "RTNet":
        model = create_model(data, data2)
    elif FLAGS.model == "CTNet":
        model = create_model(data)
    else:
        print("None of the constructed model!")
        
    ##set loss weight
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = ['mse','binary_crossentropy']
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['acc'],loss_weights=[ 1., FLAGS.loss_weight])
    
    if Flags.pretrained == 'True':
        model.load_weight(model_weights)
    
    return model, optimizer, loss
    
#### train function (version 2)####
def train():
    #### import training data and validatoin data(option)
    
    
    '''
    #### eager mode
    if FLAGS.mode == "eager_mode":
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        
        for epoch in range(1, FLAGS.max_epoch + 1):
            for batch in len(train_dataset)//FLAGS.batch_size:
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss) 
    '''
        
    else:
        sample_count = len(data)
        epochs = FLAGS.max_epoch
        warmup_epoch = int(0.2*epochs)
        total_steps = int(epochs * sample_count / FLAGS.batch_size)
        warmup_steps = int(warmup_epoch * sample_count / FLAGS.batch_size)
        
        warm_up_lr = utils.WarmUpCosineDecayScheduler(learning_rate_base=FLAGS.learning_rate,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)
        
        callback = [
            keras.callbacks.ModelCheckpoint(
                "%s.h5"%model_weights, save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=20, min_lr=1e-9
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
            warm_up_lr
            ]

        start_time = time.time()
        history = model.fit(train_dataset,
                            epochs=FLAGS.max_epoch,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        end_time = time.time() - start_time
        print(f'Total Training Time: {end_time}')
            

#########################

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
