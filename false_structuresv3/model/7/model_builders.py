import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras import regularizers

def build_model_fc3(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
     
    model=Sequential()
    model.add(Conv2D(filters=24, kernel_size=5, strides=(1,1), padding='same', activation=act, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(1,1), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Flatten())	
    model.add(Dense(10))
    model.add(Dense(output_shape, activation=final_act)) 

    return model

def build_model_fc2(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
    bias = arguments['bias_initializer'] 

    model=Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(8,activation = act, use_bias=True)) #, bias_initializer= bias))#, activity_regularizer = regularizers.l1(0.01)))
    model.add(Dense(output_shape, activation=final_act, use_bias=True))#, activity_regularizer=regularizers.l1(0.01)))

    return model

def build_model_fc2_cheat(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
    bias = arguments['bias_initializer'] 
    weights_0 = np.matrix.transpose(np.concatenate(([np.ones(32*32)],np.zeros([31,32*32])), axis =0))
    bias_0 = np.zeros(32)
    bias_1 = np.ones(1)
    bias_1 = -96*bias_1
    weights_1 = np.zeros([31,1])
    weights_1 = np.concatenate((np.ones([1,1]),weights_1), axis = 0)

    model=Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(32,activation = act, weights =[weights_0,bias_0])) 
    model.add(Dense(output_shape, activation=final_act))#, weights =[weights_1,bias_1]))

    return model

def build_model_cnn2(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
     
    model=Sequential()
    model.add(Conv2D(filters=24, kernel_size=2, strides=(2,2), padding='same', activation=act, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(Conv2D(filters=48, kernel_size=2, strides=(2,2), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Flatten())	
    model.add(Dense(10))
    model.add(Dense(output_shape, activation=final_act)) 

    return model


