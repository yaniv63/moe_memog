# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:42:11 2016

@author: yaniv
"""
from keras.layers import Dense,Input,merge, Convolution2D, LeakyReLU, MaxPooling2D, Dropout, Flatten,Layer,Dot,Concatenate,concatenate,dot
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.backend as K
import numpy as np

# def expert_model(input_shape, index,params):
#     input = Input(shape=input_shape,name='input{}'.format(index))
#     dense1 = Dense(12,activation='relu',name='exp{}_dense1'.format(index))(input)
#     dense2 = Dense(12,activation='relu',name='exp{}_dense2'.format(index))(dense1)
#     d2 =  Dropout(rate=0.5)(dense2)
#     output = Dense(1,activation='sigmoid',name ='exp{}_output'.format(index))(d2)
#     model = Model(inputs=input,outputs=output)
#     return model
#
# def simple_model(params):
#     model = Sequential()
#     model.add(Dense(12, kernel_initializer="glorot_normal", activation="relu", input_dim=28))
#     #model.add(Dropout(rate=0.5))
#     model.add(Dense(12, kernel_initializer="glorot_normal", activation="relu"))
#     model.add(Dropout(rate=0.5))
#     model.add(Dense(1, kernel_initializer="glorot_normal", activation="sigmoid"))
#     return model




def expert_model(input_shape, index,params):
    # input = Input(shape=input_shape,name='input{}'.format(index))
    # dense1 = Dense(params['nn_layer1'],activation='relu',name='exp{}_dense1'.format(index),kernel_initializer=params['w_init'])(input)
    # d1 =  Dropout(rate=params['dropout1'])(dense1)
    # dense2 = Dense(params['nn_layer2'],activation='relu',name='exp{}_dense2'.format(index),kernel_initializer=params['w_init'])(d1)
    # d2 =  Dropout(rate=params['dropout2'])(dense2)
    # output = Dense(1,activation='sigmoid',name ='exp{}_output'.format(index),kernel_initializer=params['w_init'])(d2)
    # model = Model(inputs=input,outputs=output)


    model = Sequential()
    model.add(Dense(params['nn_layer1'], activation="relu", input_dim=14,name='exp{}_dense1'.format(index),kernel_initializer=params['w_init']))
    model.add(Dropout(rate=params['dropout1']))
    model.add(Dense(params['nn_layer2'], activation="relu",name='exp{}_dense2'.format(index),kernel_initializer=params['w_init']))
    model.add(Dropout(rate=params['dropout2']))
    model.add(Dense(1, activation="sigmoid",name ='exp{}_output'.format(index),kernel_initializer=params['w_init']))
    return model

def simple_model(params):
    model = Sequential()
    model.add(Dense(params['nn_layer1'], kernel_initializer=params['w_init'], activation="relu", input_dim=28))
    model.add(Dropout(rate=params['dropout1']))
    model.add(Dense(params['nn_layer2'], kernel_initializer=params['w_init'], activation="relu"))
    model.add(Dropout(rate=params['dropout2']))
    model.add(Dense(1, kernel_initializer=params['w_init'], activation="sigmoid"))
    return model


# def n_parameters_combined_model(input_shape,n=2):
#     denses = []
#     denses2 = []
#     data = []
#
#     for i in range(n):
#         data.append(Input(shape=input_shape, name='input{}'.format(i)))
#         denses.append(Dense(12, activation='relu', name='exp{}_dense1'.format(i))(data[i]))
#         denses2.append(Dense(12, activation='relu', name='exp{}_dense2'.format(i))(denses[i]))
#
#     params = Concatenate(axis=1)(denses2)
#     params_dense = Dense(6, activation='relu', name='params_dense')(params)
#     d2 = Dropout(rate=0.5)(params_dense)
#     out = Dense(1, activation='sigmoid')(d2)
#     model = Model(inputs=data, outputs=out)
#     return model
#
#
#
# def n_experts_combined_model(input_shape,params, n=2):
#     decisions = []
#     denses = []
#     denses2 = []
#     drops = []
#     drops2= []
#     data = []
#
#     for i in range(n):
#         data.append(Input(shape=input_shape, name='input{}'.format(i)))
#         denses.append(Dense(params['nn_layer1'],kernel_initializer=params['w_init'], activation='relu', name='exp{}_dense1'.format(i))(data[i]))
#         drops.append(Dropout(rate=params['dropout1'])(denses[i]))
#         denses2.append(Dense(params['nn_layer2'],kernel_initializer=params['w_init'] ,activation='relu', name='exp{}_dense2'.format(i))(drops[i]))
#         drops2.append(Dropout(rate=params['dropout2'])(denses2[i]))
#         decisions.append(Dense(1, activation='sigmoid',kernel_initializer=params['w_init'], name='exp{}_output'.format(i))(drops2[i]))
#
#     merged_decisions = Concatenate(axis=1)(decisions)
#
#     gate_input = Concatenate(axis=1)(data)
#     gate_dense = Dense(params['nn_gate'], activation='relu',kernel_initializer=params['w_init'], name='dense1_gate')(gate_input)
#     gate_dense2 = Dense(params['nn_gate'], activation='relu',kernel_initializer=params['w_init'], name='dense2_gate')(gate_dense)
#     d2 =  Dropout(rate=params['dropout2'])(gate_dense2)
#     coefficients = Dense(n, activation='softmax',kernel_initializer=params['w_init'], name='out_gate')(d2)
#
#     weighted_prediction = Dot(axes=1)([coefficients, merged_decisions])
#     model = Model(inputs=data, outputs=weighted_prediction)
#     return model
#
#
# def n_experts_combined_model_gate_parameters(input_shape, n=2):
#
#     decisions = []
#     denses = []
#     denses2 = []
#     drops = []
#     data = []
#
#     for i in range(n):
#         data.append( Input(shape=input_shape, name='input{}'.format(i)))
#         denses.append(Dense(12, activation='relu', name='exp{}_dense1'.format(i))(data[i]))
#         denses2.append(Dense(12, activation='relu', name='exp{}_dense2'.format(i))(denses[i]))
#         drops.append(Dropout(rate=0.5)(denses2[i]))
#         decisions.append(Dense(1, activation='sigmoid', name='exp{}_output'.format(i))(drops[i]))
#
#     merged_decisions = Concatenate(axis=1)(decisions)
#
#     #gate
#     gate_input = Concatenate(axis=1)(denses2)
#     gate_dense = Dense(6,activation='relu',name='dense1_gate')(gate_input)
#     d2 =  Dropout(rate=0.5)(gate_dense)
#     coefficients = Dense(n, activation='softmax', name='out_gate')(d2)
#
#     weighted_prediction = Dot(axes=1)([coefficients, merged_decisions])
#     model = Model(inputs=data, outputs=weighted_prediction)
#     return model

# def moe_expert(input_shape,i,params):
#     exp = Sequential(name="exp{}".format(i))
#     exp.add(Dense(params['nn_layer1'], kernel_initializer=params['w_init'], activation='relu',input_shape=input_shape,
#                   name='exp{}_dense1'.format(i)))
#     exp.add(Dropout(rate=params['dropout1']))
#     exp.add(Dense(params['nn_layer2'], kernel_initializer=params['w_init'], activation='relu',
#                   name='exp{}_dense2'.format(i)))
#     exp.add(Dropout(rate=params['dropout2']))
#     exp.add(Dense(1, activation='sigmoid', kernel_initializer=params['w_init'], name='exp{}_output'.format(i)))
#     return exp

def n_experts_combined_model_b(input_shape,params, n=2):
    decisions = []
    experts = []
    data = []

    for i in range(n):
        experts.append(expert_model(input_shape,i,params))
        data.append(experts[i].input)
        decisions.append(experts[i].output)

    merged_decisions = concatenate(decisions,axis=1)

    gate_input = concatenate(data,axis=1)
    gate_dense = Dense(params['nn_gate'], activation='relu', name='dense1_gate', kernel_initializer=params['w_init'])(gate_input)
    gate_dense2 = Dense(params['nn_gate'], activation='relu', name='dense2_gate', kernel_initializer=params['w_init'])(gate_dense)
    d2 =  Dropout(rate=params['dropout2'])(gate_dense2)
    coefficients = Dense(n, activation='softmax', name='out_gate', kernel_initializer=params['w_init'])(d2)

    weighted_prediction = dot([coefficients, merged_decisions],axes=1,name='main_output')
    model = Model(inputs=data, outputs=weighted_prediction)
    return model



def check_params_model(input_shape,params, n=2):
    model = n_experts_combined_model_b(input_shape,params, n)
    check_model = Model(
        inputs=model.inputs,
        outputs=[ model.output,
                  model.get_layer('out_gate').output,
                  model.get_layer('exp0_output').output,
                  model.get_layer('exp1_output').output
                  ]
    )
    return check_model

def multilabel_model(input_shape,params, n=2):
    model = n_experts_combined_model_b(input_shape,params, n)
    multilabel = Model(
        inputs=model.inputs,
        outputs=[model.output,
                 model.get_layer('exp0_output').output,
                 model.get_layer('exp1_output').output
                 ]
    )
    return multilabel

# params = {
#     'optimizer' : 'adam',
#     'nn_layer1' : 24,
#     'nn_layer2' : 24,
#     'dropout1' : 0.8,
#     'dropout2': 0.8,
#     'epoch_num' : 500,
#     'w_init' : 'glorot_uniform',
#     'nn_gate' : 10
# }
# a = n_experts_combined_model_b(params=params,input_shape=14)
# from keras.utils import plot_model
# plot_model(a,to_file='n_experts_combined_model_b.png',show_layer_names=True,show_shapes=True)


