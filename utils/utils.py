#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
### tf.to_int64 (TF1.X) --> tf.compat.v1.to_int64 (TF2.X)
#def cal_rsquared(label, pred, loss):
  #unexplained_loss = tf.reduce_sum(tf.square(tf.subtract(tf.compat.v1.to_int64(label),pred)))
  #r_2 = tf.subtract(1, tf.divide(unexplained_loss,tf.reduce_sum(loss)))  
  #return r_2
  

def cal_rsquared(label, pred):
    
    residual = tf.reduce_sum(tf.square(tf.subtract(label,pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(label, tf.reduce_mean(label))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def shuffle_data(data, labels):
  """ Shuffle data and labels.
      shuffled data, label and shuffle indices
  """
  idx = np.arange(len(labels))
  np.random.shuffle(idx)
  return data[idx], labels[idx], idx

def col_append(columns,num_dup):
    '''
    return all columns, all columns deleted categroical 
    '''
    for i in range(num_dup):
        if i==0:
            column=list(columns)
        else:
            column=column+list(map(lambda x: x+'_'+str(i),columns))
    return column

def next_time_pred(df,target,date):
    df['new_target']=df.groupby(date)[target].shift(-1)
    df=df.dropna(subset=['new_target'])
    return df

def time_series_trans(pid,date,new_target,features,time_length):
    total_list=[]
    for pno2 in df[pid].unique():    
        print(str(list(df[pid].unique()).index(pno2)+1)+ '/'+ str(len(list(df[pid].unique())))+'........\n')
        df=temp[pno2][features]
        for date in df[date].unique():   
            print(str(list(df[date].unique()).index(date)+1)+ '/'+ str(len(list(df[date].unique())))+'........\n')
            date_df=df[df[date]==date].reset_index(drop=True)
            if len(date_df)<=1:
                print('Error: Time length of record of %s is less than 2 in the patient ID %s',(str(pno2),str(date)))
            for num in range(len(date_df)):
                #situation 1, the first record of the day
                if num == 0:
                    temp_list=list(pd.concat([date_df.drop([new_target],axis=1)[0:1]]*time_length, ignore_index=True).values.reshape(-1))
                    temp_list.append(date_df[new_target][num])             
                    if total_list==[]:                
                        total_list=[temp_list]
                    else:
                        total_list.append(temp_list)
                #situation 2, the transformation which have duplicate first record
                elif 1<=num<=time_length:   
                    temp_list=list(pd.concat([date_df.drop([new_target],axis=1)[0:1]]*(time_length-num), ignore_index=True).values.reshape(-1))
                    temp_list.extend(date_df.drop([new_target],axis=1)[1:num+1].values.reshape(-1))
                    temp_list.append(date_df[new_target][num])
                    total_list.append(temp_list)
                #situation 3, no duplicate record 
                else:
                    temp_list=list(date_df.drop([new_target],axis=1)[num-(time_length-1):num+1].values.reshape(-1))
                    temp_list.append(date_df[new_target][num])
                    total_list.append(temp_list)
        print('-'*30)            
        small_df=pd.DataFrame(total_list,columns=columnnn)           
        small_df.to_csv('new_fixed_length_partof.csv')
        
    print('Final csv file saved.')
    final_df=pd.DataFrame(total_list,columns=columnnn)
    final_df.to_csv('new_fixed_length_new.csv')
    return final_df

#####################################################################################################

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
