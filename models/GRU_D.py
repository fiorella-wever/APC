"""Implementation of GRU-D model.

This implementation is based on and adapted from
https://github.com/PeterChe1990/GRU-D

Which is published unter the MIT licence.
"""


from __future__ import absolute_import, division, print_function

import keras.backend as K
from keras.layers import Activation, Dense, Dropout, Input, Masking
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.models import load_model, Model
from keras.regularizers import l2
from keras.utils.generic_utils import custom_object_scope
import tensorflow as tf
import tensorflow.keras
import numpy as np

from nn_utils.grud_layers import GRUD
from nn_utils.layers import ExternalMasking


def create_grud_model(input_dim, aux_dim, hidden_neurons, dropout_rate, recurrent_dropout_rate,
                      output_dim, predefined_model=None, **kwargs):
    
    if (predefined_model is not None
            and predefined_model in _PREDEFINED_MODEL_LIST):
        for c, v in _PREDEFINED_MODEL_LIST[predefined_model].items():
            kwargs[c] = v
    
    # Input
    input_x = Input(shape=(None, input_dim))
    input_m = Input(shape=(None, input_dim))
    input_s = Input(shape=(None, 1))
    aux_input = Input(shape=(aux_dim,))
    length = Input(shape=(1,))
    input_list = [input_x, input_m, input_s, aux_input, length]
    input_x = ExternalMasking()([input_x, input_m])
    input_s = ExternalMasking()([input_s, input_m])
    input_m = Masking()(input_m)
    # GRU layers
    grud_layer = GRUD(units=hidden_neurons,
                      return_sequences=True,
                      activation='tanh',
                      dropout=dropout_rate,
                      recurrent_dropout=recurrent_dropout_rate,
                      **kwargs)

    x = grud_layer([input_x, input_m, input_s])
    
    def slice_last_non_zero_hidden(x, l):
        l = tf.cast(l, "int32")
        indices = tf.stack([tf.range(tf.shape(l)[0]), tf.reshape(l, [-1])],axis=1)
        x_out = tf.gather_nd(x, indices)
        return x_out
    
    x = Lambda(lambda x: slice_last_non_zero_hidden(x[0], x[1]))([x,length])

    main_input = Concatenate(axis=1)([x, aux_input])
    
    x_out = Dense(output_dim, activation="softmax")(main_input)
    output_list = [x_out]

    model = Model(inputs=input_list, outputs=output_list)
    
    return model


def load_grud_model(file_name):
    from nn_utils import _get_scope_dict
    with custom_object_scope(_get_scope_dict()):
        model = load_model(file_name)
    return model


_PREDEFINED_MODEL_LIST = {
    'GRUD': {
        'x_imputation': 'zero',
        'input_decay': 'exp_relu',
        'hidden_decay': 'exp_relu',
        'feed_masking': True,
    },
    'GRUmean': {
        'x_imputation': 'zero',
        'input_decay': None,
        'hidden_decay': None,
        'feed_masking': False,
    },
    'GRUforward': {
        'x_imputation': 'forward',
        'input_decay': None,
        'hidden_decay': None,
        'feed_masking': False,
    },
    'GRUsimple': {
        'x_imputation': 'zero',
        'input_decay': None,
        'hidden_decay': None,
        'feed_masking': True,
    },
}

