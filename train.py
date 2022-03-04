import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import transformer_encoder
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
parser.add_argument('--model', default='CTNet', help='Model name: CTNet')
parser.add_argument('--pretrained', type=str_to_bool, default=False, help='boolean value')
parser.add_argument('--mode', default='training_mode', help='traing mode[default: training_mode]')
parser.add_argument('--val_data', type=str_to_bool, default=False, help='boolean value')
#parser.add_argument('--num_data', type=int, default='10', help='number of balanced data')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_trans_enc', type=int, default=2, help='Number of transformer encoder [default: 2]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]') 
parser.add_argument('--loss1', default='mse', help='Loss1 [default: mse]')
parser.add_argument('--loss2', default='mse', help='Loss2 [default: mse]')
parser.add_argument('--loss_weight', type=int, default=1, help='Initial loss weight [default: 2]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
## transformer encoder default setting
parser.add_argument('--time_steps', type=int, default=8, help='Number of time steps [default: 8]')
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
#os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # backup of model definition
#os.system('cp train.py %s' % (LOG_DIR)) # backup of train process
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
        print("the data file can't be processed!")
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
    



def setup_model(data, data2=None):
    #look_ahead_mask = transformer_encoder.create_look_ahead_mask(time_steps= data.shape[1])
    
    if FLAGS.model == "RTNet":
        model = MODEL.create_model(data, data2, FLAGS.num_trans_enc)
    elif FLAGS.model == "CTNet":
        model = MODEL.create_model(data, FLAGS.num_trans_enc)
    else:
        print("None of the constructed model!")
        
    ##set loss weight
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [FLAGS.loss1, FLAGS.loss2]
    
    model.compile(loss=loss, optimizer=optimizer,metrics=['acc'],loss_weights=[ 1., FLAGS.loss_weight])
    
    if FLAGS.pretrained == 'True':
        model.load_weight(model_weights)
    
    return model, optimizer, loss
    
#### shuffle function

    
#### train function (version 2)####
def train():
    #### import training data and validatoin data(option)
    data = pd.read_csv('final_processed_partof.csv',index_col=[0])
    data = data.sample(frac=1).reset_index(drop=True)

    #data.select_dtypes(exclude=['object'])
    ## excluded data which don't belong to  features
    train_feature=[s for s in data.columns if 'County' not in s and 'City' not in s and 'Site Num' not in s and 'Date Local' not in s]
    
    ## shuffle training data
    if FLAGS.model == 'RTNet':
        data2 = pd.read_csv('.csv',index_col=[0])
        data2 = data2.sample(frac=1).reset_index(drop=True)
       
    if FLAGS.val_data == True:
        val_data = pd.read_csv('.csv',index_col=[0])
        if FLAGS.model == 'RTNet':
            val_data2 = pd.read_csv('.csv',index_col=[0])
        
    else:
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)    
        
        if FLAGS.model == 'RTNet':
            temp = pd.concat([data, data2],axis=1)
            train_temp, val_temp = train_test_split(temp, test_size=0.2, random_state=42)
            train_data = train_temp.iloc[:, :len(data)]
            train_data2 = train_temp.iloc[:, len(data):]            
            val_data = val_temp.iloc[:, :len(val_data)]
            val_data2 = val_temp.iloc[:, len(val_data):]
            
    #### exact target from data (Due to two-task training, so there were target data in the last two columns of the data)
    if FLAGS.model == 'CTNet':
        train_label1, train_label2 = tf.convert_to_tensor(train_data.iloc[:, -2:-1]), tf.convert_to_tensor(train_data.iloc[:, -1:])
        val_label1, val_label2 = tf.convert_to_tensor(val_data.iloc[:, -2:-1]), tf.convert_to_tensor(val_data.iloc[:, -1:])
        print(len(train_data.columns))
        train_data=np.dstack(np.split(train_data.iloc[:,:-2][train_feature].values, FLAGS.time_steps, axis = 1))
        train_data=tf.convert_to_tensor(np.moveaxis(train_data, 1, 2))       
        val_data=np.dstack(np.split(val_data.iloc[:,:-2][train_feature].values, FLAGS.time_steps, axis = 1))
        val_data=tf.convert_to_tensor(np.moveaxis(val_data, 1, 2))         
        
    elif FLAGS.model == 'RTNet':
        train_label1, train_label2 = tf.convert_to_tensor(train_data.iloc[:, -2:-1]), tf.convert_to_tensor(train_data.iloc[:, -1:])
        val_label1, val_label2 = tf.convert_to_tensor(val_data.iloc[:, -2:-1]), tf.convert_to_tensor(val_data.iloc[:, -1:])
        #train_data = train_data.iloc[:, :-2]
        #train_data2 = train_data2.iloc[:, :-2]            
        #val_data = val_data.iloc[:, :-2]
        #val_data2 = val_data2.iloc[:, :-2]
        
        # filtered Identification columns
        train_data=np.dstack(np.split(train_data.iloc[:,:-2][train_feature].values, FLAGS.time_steps, axis = 1))
        train_data=tf.convert_to_tensor(np.moveaxis(train_data, 1, 2))
        train_data2=np.dstack(np.split(train_data2.iloc[:,:-2][train_feature].values, FLAGS.time_steps, axis = 1))
        train_data2=tf.convert_to_tensor(np.moveaxis(train_data2, 1, 2)) 
        
        val_data=np.dstack(np.split(val_data.iloc[:,:-2][train_feature].values, FLAGS.time_steps, axis = 1))
        val_data=tf.convert_to_tensor(np.moveaxis(val_data, 1, 2)) 
        val_data2=np.dstack(np.split(val_data2.iloc[:,:-2][train_feature].values, FLAGS.time_steps, axis = 1))
        val_data2=tf.convert_to_tensor(np.moveaxis(val_data2, 1, 2)) 
        
    # Constructing model    
    if FLAGS.model == 'CTNet':
        model, optimizer, loss = setup_model(train_data)
    if FLAGS.model == 'RTNet':
        model, optimizer, loss = setup_model(train_data, train_data2)  
    '''
    #### eager mode
    if FLAGS.mode == "eager_mode":
        avg_loss1 = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_loss2 = tf.keras.metrics.Mean('loss', dtype=tf.float32)        
        avg_val_loss1 = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        avg_val_loss2 = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        
        for epoch in range(1, FLAGS.max_epoch + 1):
            for batch in len(train_data)//FLAGS.batch_size:
                with tf.GradientTape() as tape:
                    if FLAGS.model == 'CTNet':
                        output1, output2 = model(train_data, training=True)
                    elif FLAGS.model == 'RTNet':
                        output1, output2 = model([train_data, train_data2], training=True)
                    regularization_loss1, regularization_loss2 = model.losses
                    regularization_loss1, regularization_loss2 = tf.reduce_sum(regularization_loss1), tf.reduce_sum(regularization_loss2)
                    pred_loss = []
                    for output_1, output_2, label_1, label_2, loss1, loss2 in zip(output1, output2, label1, label2, loss[0], loss[1]):
                        pred_loss1.append(loss1(label_1, output_1))
                        pred_loss2.append(loss2(label_2, output_2))
                    total_loss1 = tf.reduce_sum(pred_loss1) + regularization_loss1
                    total_loss2 = tf.reduce_sum(pred_loss2) + regularization_loss2

                grads1 = tape.gradient(total_loss1, model.trainable_variables)
                grads2 = tape.gradient(total_loss2, model.trainable_variables)
                
                optimizer.apply_gradients(
                    zip(grads1, model.trainable_variables))
                    
                logging.info("{}_train_loss1_{}, {}, {}".format(
                    epoch, batch, total_loss1 .numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss1))))
                avg_loss1.update_state(total_loss1) 
                
                optimizer.apply_gradients(
                    zip(grads2, model.trainable_variables))                    

                logging.info("{}_train_loss1_{}, {}, {}".format(
                    epoch, batch, total_loss1 .numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss1))))
                avg_loss2.update_state(total_loss1) 
                
                ## validation dataset
                if FLAGS.model == 'CTNet':
                    output1, output2 = model(val_data)
                elif FLAGS.model == 'RTNet':
                    output1, output2 = model([val_data, val_data2])
                regularization_loss1, regularization_loss2 = model.losses
                regularization_loss1, regularization_loss2 = tf.reduce_sum(regularization_loss1), tf.reduce_sum(regularization_loss2)
                pred_loss = []
                for output_1, output_2, label_1, label_2, loss1, loss2 in zip(output1, output2, label1, label2, loss[0], loss[1]):
                    pred_loss1.append(loss1(label_1, output_1))
                    pred_loss2.append(loss2(label_2, output_2))
                total_loss1 = tf.reduce_sum(pred_loss1) + regularization_loss1
                total_loss2 = tf.reduce_sum(pred_loss2) + regularization_loss2
                
                logging.info("{}_train_loss1_{}, {}, {}".format(
                    epoch, batch, total_loss1 .numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss1))))
                avg_val_loss1.update_state(total_loss1) 
                
                logging.info("{}_train_loss1_{}, {}, {}".format(
                    epoch, batch, total_loss2 .numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss2))))
                avg_val_loss2.update_state(total_loss2) 
                
            # print result     
            logging.info("{}, train loss 1: {}, train loss 2: {}, val loss 1: {}, val loss 2: {}".format(
                epoch,
                avg_loss1.result().numpy(),
                avg_loss2.result().numpy(),
                avg_val_loss1.result().numpy(),
                avg_val_loss2.result().numpy())

            avg_loss1.reset_states()
            avg_loss2.reset_states()
            avg_val_loss1.reset_states()
            avg_val_loss2.reset_states()
            model.save_weights(
                'checkpoints/train_{}.tf'.format(epoch))

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
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "%s.h5"%model_weights, save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=20, min_lr=1e-9
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
            warm_up_lr
            ]

        start_time = time.time()
        if FLAGS.model == 'RTNet':
            history = model.fit([train_data, train_data2],[train_label1, train_label2],
                                epochs = FLAGS.max_epoch,
                                batch_size = FLAGS.batch_size,
                                callbacks = callbacks,
                                validation_data = ([val_data, val_data2], [val_label1, val_label2])
                               )
        elif FLAGS.model == 'CTNet':
            history = model.fit(train_data,[train_label1, train_label2],
                                epochs = FLAGS.max_epoch,
                                batch_size = FLAGS.batch_size,
                                callbacks = callbacks,
                                validation_data = (val_data, [val_label1, val_label2])
                                )
        else:
            print("None of the 'RTNet' or 'CTNet' model!")
        
        end_time = time.time() - start_time
        print(f'Total Training Time: {end_time}')
            

#########################

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
