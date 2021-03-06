#############
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../transformer_encoder'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import transformer_encoder as te
from transformer_encoder import *
import utils
from keras import optimizers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector

#adam = optimizers.Adam(lr=0.001)
#################################################################################################
#### merge all variable and training together
#features=np.array(total_X_train_cb[0]).shape[2]
#################################################################################################

def create_model(data_df, num_trans_enc, d_k=64, d_v=64, ff_dim=256, n_heads=2):
    
    
    batch_size = data_df.shape[0]
    time_steps = data_df.shape[1] 
    features = data_df.shape[2]  

    '''Initialize time and transformer layers'''    
    time_embedding = te.Time2Vector(time_steps)
    num_te=[]
    for i in range(num_trans_enc):
        num_te.append(te.TransformerEncoder(d_k, d_v, ff_dim, n_heads, mask=create_look_ahead_mask(time_steps), dropout=0.2))
    '''Construct model'''
    in_seq = Input(shape=(time_steps, features))
    x2 = time_embedding(in_seq)
    x2 = Concatenate(axis=-1)([in_seq, x2])

    for i in range(num_trans_enc):
        x2 = num_te[i]((x2, x2, x2))

    x = GlobalAveragePooling1D()(x2)

    out1 = Dense(1, activation='linear',name='reg')(x)

    out2 = Dense(1, activation='linear',name='reg2')(x)
    #out2 = Dense(1, activation='sigmoid',name='binary_class')(x)        
      
    model = Model(inputs=in_seq, outputs=[out1,out2])

    #model.compile(loss=['mse','binary_crossentropy'], optimizer=adam,metrics=['acc'],loss_weights=[ 1., 100.])

    return model 

#############
if __name__=='__main__':
    batch_size = 64
    d_k = 64
    d_v = 64
    ff_dim = 256
    seq_len=8
    n_heads=2
    look_ahead_mask = transformer_encoder.create_look_ahead_mask(seq_len)
