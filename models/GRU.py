from keras.models import Model, Input
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Concatenate
from keras import optimizers
from keras_layer_normalization import LayerNormalization

#general
def GRU_model(x_length, n_features, n_aux, n_classes, n_neurons, learning_rate, dropout_rate, recurrent_dropout, loss_type):
    
    input1 = Input(shape=(x_length, n_features)) 
    x = GRU(n_neurons, activation='relu', dropout=dropout_rate, recurrent_dropout=recurrent_dropout, return_sequences=False)(input1)
    x = LayerNormalization()(x)
    
    aux_input = Input(shape=(n_aux,))    
    main_input = Concatenate(axis=1)([x, aux_input])
 
    output = Dense(n_classes, activation='softmax')(main_input)
    model = Model(inputs=[input1, aux_input], outputs=output)
    adam_optim = optimizers.Adam(lr=learning_rate) 
    model.compile(loss=loss_type, optimizer=adam_optim)
    
    return model