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
#########################################################################################################
#features=np.array(total_X_train_time[0]).shape[2]
#risk_features=np.array(total_X_train_risk[0]).shape[2]
##########################################################################################################

def create_model(data_invariant, data_time, output_bias=None):
  batch_size = data_invariant.shape[0]
  time_steps = data_invariant.shape[1] 
  inv_features = data_invariant.shape[2]
  time_features = data_time.shape[2]  

  
  risk_seq = Input(shape=(time_steps, inv_features))
  x1 = Conv1D(filters=64, kernel_size=1,use_bias=False)(risk_seq)
  x1 = BatchNormalization()(x1)
  x1 = ReLU()(x1)
  x1 = Dropout(0.2)(x1)

  x1 = Conv1D(filters=inv_features, kernel_size=1,use_bias=False)(x1)
  x1 = BatchNormalization()(x1)
  x1 = ReLU()(x1)
  x1 = Dropout(0.2)(x1)
     
  '''Initialize time and transformer layers'''
  time_embedding = te.Time2Vector(time_steps)
  #attn_layer1 = te.TransformerEncoder(d_k, d_v, ff_dim, n_heads, mask=look_ahead_mask, dropout=0.2)
  #attn_layer2 = te.TransformerEncoder(d_k, d_v, ff_dim, n_heads, mask=look_ahead_mask, dropout=0.2)
  #attn_layer3 = te.TransformerEncoder(d_k, d_v, ff_dim, n_heads, mask=look_ahead_mask, dropout=0.2)
  #attn_layer4 = te.TransformerEncoder(d_k, d_v, ff_dim, n_heads, mask=look_ahead_mask, dropout=0.2)
  num_te=[]
  for i in range(num_trans_enc):
      num_te.append(te.TransformerEncoder(d_k=d_k, d_v=d_v, ff_dim=ff_dim, n_heads=n_heads, mask=look_ahead_mask, dropout=0.2))
  '''Construct model'''
  in_seq = Input(shape=(time_steps, time_features))
  x2 = time_embedding(in_seq)
  x2 = Concatenate(axis=-1)([in_seq, x2])
  # set different number transformer encoder
  #x2 = attn_layer1((x2, x2, x2))
  #x2 = attn_layer2((x2, x2, x2))
  for i in range(num_trans_enc):
     x2 = num_te[i]((x2, x2, x2))
    
  x = Concatenate(axis=-1)([x1, x2])
  x = GlobalAveragePooling1D()(x)

  out1 = Dense(1, activation='linear',name='reg')(x)
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
    out2 = Dense(1, activation='sigmoid',name='binary_class',bias_initializer=output_bias)(x)
  else:
    out2 = Dense(1, activation='sigmoid',name='binary_class')(x)        
      
  model = Model(inputs=[risk_seq, in_seq], outputs=[out1,out2])

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
    
