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


def input_placeholder(batch_size, time_steps, features):
    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    data_pl = tf.placeholder(tf.float32, shape=(batch_size, time_steps, features))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return data_pl, labels_pl

 
def create_model(data_df, output_bias=None):
    
    batch_size = data_df.get_shape()[0].value
    time_steps = data_df.get_shape()[1].value  
    features = data_df.get_shape()[2].value  

    '''Initialize time and transformer layers'''    
    time_embedding = te.Time2Vector(time_steps)
    num_te=[]
    for i in range(num_trans_enc):
        num_te.append(te.TransformerEncoder(d_k, d_v, ff_dim, n_heads, mask=look_ahead_mask, dropout=0.2))
    '''Construct model'''
    in_seq = Input(shape=(time_steps, features))
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
      
    return out1, out2

def loss_def(pred1, pred2, label1, label2, impt_weight=100):
    label2_onehot = tf.one_hot(indices=label2, depth=2)
    loss1 = tf.losses.mean_squared_error(label1, pred1)
    loss2 = tf.losses.sigmoid_cross_entropy(label2_onehot, logit = pred2)
  
    final_loss = tf.reduce_mean(loss1 + impt_weight*loss2)
    final_loss = tf.reduce_mean(final_loss)
    return final_loss, loss1, loss2

#############
if __name__=='__main__':
    batch_size = 64
    d_k = 64
    d_v = 64
    ff_dim = 256
    seq_len=8
    n_heads=2
    look_ahead_mask = transformer_encoder.create_look_ahead_mask(seq_len)
