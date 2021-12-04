#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import keras.backend as K
from keras.models import Model, Input
from keras.layers import Masking
from keras.layers.merge import add
from keras.layers import TimeDistributed
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import GRU
from keras import optimizers
from keras_layer_normalization import LayerNormalization
from keras.layers import Concatenate
from nn_utils.grud_layers import GRUD
from nn_utils.layers import ExternalMasking
import tensorflow as tf


## Encoder Models

def GRU_layers(input_dim, hidden_neurons, aux_dim, dropout_rate, recurrent_dropout_rate, mask=False):
    
    # Input
    input1 = Input(shape=(None, input_dim)) 
    aux_input = Input(shape=(aux_dim,))
    input_list = [input1, aux_input]
    
    x = input1
    if mask:
        x = Masking(mask_value=-1)(x)
    x = GRU(hidden_neurons, activation='relu', dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, return_sequences=True)(x)
    x = LayerNormalization()(x)
        
    def slice_last(x):
        return x[..., -1, :]
    
    # get last hidden state, used for classification
    x_out = Lambda(slice_last)(x)    
        
    return input_list, x, x_out
    
def GRUD_layers(input_dim, hidden_neurons, aux_dim, dropout_rate, recurrent_dropout_rate):
    
    #Input
    input_x = Input(shape=(None, input_dim))
    input_m = Input(shape=(None, input_dim))
    input_s = Input(shape=(None, 1))
    aux_input = Input(shape=(aux_dim,))
    length = Input(shape=(1,))
    input_list = [input_x, input_m, input_s, aux_input, length]
    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)
    
    #GRU-D model
    grud_layer = GRUD(units=hidden_neurons,
                      return_sequences=True,
                      activation='tanh',
                      dropout=dropout_rate,
                      recurrent_dropout=recurrent_dropout_rate,
                      x_imputation = 'zero',
                      input_decay = 'exp_relu',
                      hidden_decay = 'exp_relu',
                      feed_masking = True)
    x = grud_layer([input_x, input_m, input_s])

    def slice_last_non_zero_hidden(x, l):
        l = tf.cast(l, "int32")  
        indices = tf.stack([tf.range(tf.shape(l)[0]), tf.reshape(l, [-1])],axis=1)
        x_out = tf.gather_nd(x, indices)
        return x_out
    
    # get last hidden state, used for classification
    x_out = Lambda(lambda x: slice_last_non_zero_hidden(x[0], x[1]))([x,length])
    
    return input_list, x, x_out
    
    
def create_APC_classifier(config, encoder, stop_APC_grad):   
    # APC reconstruction 
    # GRU or GRU-D encoder
    if encoder == "GRU":
        input_list, h, last_h = GRU_layers(config["n_features"], config["n_neurons"], config["aux_dim"], config["dropout_rate"], config["recurrent_dropout_rate"], mask=False)
        aux_input = input_list[1]
    elif encoder == "GRUD":
        input_list, h, last_h = GRUD_layers(config["n_features"], config["n_neurons"], config["aux_dim"], config["dropout_rate"], config["recurrent_dropout_rate"])
        aux_input = input_list[3]
    
    # if specified, keep APC weights frozen
    if stop_APC_grad:
        h = Lambda(lambda x: K.stop_gradient(x))(h)
    
    # APC reconstruction output
    output_1 = TimeDistributed(Dense(config["n_features"]))(h)
    
    # classification 
    classifier_input = Concatenate(axis=1)([last_h, aux_input])
    # classifier output
    output_2 = Dense(config["n_classes"], activation="softmax")(classifier_input)
    
    output_list = [output_1, output_2]
     
    model = Model(inputs=input_list, outputs=output_list)
 
    adam_optim = optimizers.Adam(lr=config["learning_rate"])
    model.compile(loss=[config["l1_type"], config["l2_type"]], loss_weights = [config["l1"], config["l2"]], optimizer=adam_optim, metrics={'dense_3': config["evaluation_metric"]})
    
    return model