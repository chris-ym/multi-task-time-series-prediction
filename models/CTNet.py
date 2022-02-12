#############
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import transformer_encoder as te
import utils
from keras import optimizers
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector

adam = optimizers.Adam(lr=0.001)
#################################################################################################
#### merge all variable and training together
features=np.array(total_X_train_cb[0]).shape[2]
#################################################################################################

def create_model(output_bias=None):

    '''Initialize time and transformer layers'''    
    time_embedding = te.Time2Vector(seq_len)
    num_te=[]
    for i in range(num_trans_enc):
        num_te.append(te.TransformerEncoder(d_k, d_v, ff_dim, n_heads, mask=look_ahead_mask, dropout=0.2))
    '''Construct model'''
    in_seq = Input(shape=(seq_len, features))
    x2 = time_embedding(in_seq)
    x2 = Concatenate(axis=-1)([in_seq, x2])
    
    for i in range(num_trans_enc):
       x2 = num_te[i]((x2, x2, x2))
    
    x = GlobalAveragePooling1D()(x2)

    out1 = Dense(1, activation='linear',name='reg')(x)
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        out2 = Dense(1, activation='sigmoid',name='binary_class',bias_initializer=output_bias)(x)
    else:
        out2 = Dense(1, activation='sigmoid',name='binary_class')(x)

    model = Model(inputs=in_seq, outputs=[out1,out2])

    model.compile(loss=['mse','binary_crossentropy'], optimizer=adam,metrics=['acc'])

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
