"""
This is a modification to Demo_start_training_proceedure.py where it repeats 
the code until it finds a model with a false structure

This file read a `config.yml` file and train a network based on the settings
in that file
"""

import yaml
import shutil
import uuid
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from nn_tools import read_count
import tensorflow as tf
from Data_loader_stripe import Data_loader_stripe_train, Data_loader_stripe_test
import model_builders as mb
from SaveWeights import MyCallback
import os
from os.path import join
import matplotlib.pyplot as plt;
import numpy as np;

def data_selector(data_name, arguments):
    """ Select a data loader based on `data_name` (str).
Arguments
---------
data_name (str): Name of the data loader
arguments (dict): Dictionary given to the constructor of the data loader

Returns
-------
Data loader with name `data_name`. If not found, an error message is printed
and it returns None.
"""
    if data_name.lower() == "stripe_train":
        return Data_loader_stripe_train(arguments)
    elif data_name.lower() == "stripe_test":
        return Data_loader_stripe_test(arguments)
    else:
        print('Error: Could not find data loader with name %s' % (data_name))
        return None;

def model_selector(model_name, input_shape, output_shape, arguments):
    """ Select a model (network) based on `model_name` (str).
Arguments
---------
model_name (str): Name of the model loader
input_shape (list): List of integers specifying dimensions
output_shape (list): List of integers specifying dimensions
arguments (dict): Arguments to the model function

Returns
-------
Keras model
"""
    if model_name.lower() == "fc3":
        return mb.build_model_fc3(input_shape, output_shape, arguments)
    elif model_name.lower() == "fc2":
        return mb.build_model_fc2(input_shape, output_shape, arguments)
    elif model_name.lower() == "fc2_cheat":
        return mb.build_model_fc2_cheat(input_shape, output_shape, arguments)
    elif model_name.lower() == "cnn2":
        return mb.build_model_cnn2(input_shape, output_shape, arguments)
    else:
        print('Error: Could not find model with name %s' % (model_name))
        return None;

if __name__ == "__main__":
    """Train a full model based on the settings in `config.yml`"""
    # Turn off unnessesary warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Load configuration file
    with open('config.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);

    # Set up computational resource 
    use_gpu = cgf['COMPUTER_SETUP']['use_gpu']
    print("""\nCOMPUTER SETUP
Use gpu: {}""".format(use_gpu))
    if use_gpu:
        compute_node = cgf['COMPUTER_SETUP']['compute_node']
        os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
        print('Compute node: {}'.format(compute_node))
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

    # Turn on soft memory allocation
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.compat.v1.Session(config=tf_config)
    #K.v1.set_session(sess)
    
    # Load train and validatation data
    data_loader_train = data_selector(cgf['DATASET_TRAIN']['name'], cgf['DATASET_TRAIN']['arguments'])
    data_loader_validate = data_selector(cgf['DATASET_VAL']['name'], cgf['DATASET_VAL']['arguments'])
    print('\nDATASET TRAIN')
    print(data_loader_train)
    print('DATASET VALIDATION')
    print(data_loader_validate)

    train_data, train_labels = data_loader_train.load_data();
    val_data, val_labels = data_loader_validate.load_data();

    # Get input and output shape
    input_shape = train_data.shape[1:]
    output_shape = train_labels.shape[1];
    print('input_shape', input_shape)
    # Set the default precision 
    model_precision = cgf['MODEL_METADATA']['precision']
    K.set_floatx(model_precision)

    # Print model information
    model_name = cgf['MODEL']['name']
    model_arguments = cgf['MODEL']['arguments']
    print("""MODEL
name: {}
arguments:""".format(model_name))
    for key, value in cgf['MODEL']['arguments'].items():
        print("\t{}: {}".format(key,value))

    # Load model
    model = model_selector(model_name, input_shape, output_shape, model_arguments)

    # Extract training information
    loss_type = cgf['TRAIN']['loss']['type']
    optimizer = cgf['TRAIN']['optim']['type']
    batch_size = cgf['TRAIN']['batch_size']
    metric_list = list(cgf['TRAIN']['metrics'].values()) 
    shuffle_data = cgf['TRAIN']['shuffle'] 
    max_epoch = cgf['TRAIN']['max_epoch']
    stpc_type = cgf['TRAIN']['stopping_criteria']['type']
    print("""\nTRAIN
loss: {}
optimizer: {}
batch size: {}
shuffle data between epochs: {}
max epoch: {}
stopping_criteria: {}""".format(loss_type, optimizer, batch_size, shuffle_data, 
           max_epoch, stpc_type))

    callbacks = None
    if stpc_type.lower() == 'epoch' or stpc_type.lower() == 'epochs':
        callbacks = None
    elif stpc_type.lower() == "earlystopping" or stpc_type.lower() == "early_stopping":
        arguments = cgf['TRAIN']['stopping_criteria']['arguments']
        es = EarlyStopping(**arguments)
        callbacks = [es]
        for key, value in arguments.items():
            print('\t{}: {}'.format(key,value))
    else:
        print('Unknown stopping criteria')
        callbacks = None
    print('')
    # Compile model
    model.compile(optimizer = optimizer, 
                 loss = loss_type,
                 metrics = metric_list)

    # Model metadata
    save_final_model = cgf['MODEL_METADATA']['save_final_model']
    save_best_model = cgf['MODEL_METADATA']['save_best_model']['use_model_checkpoint']
    print("""MODEL METADATA
Precision: {}
Save final model: {}""".format(model_precision, save_final_model))
    if save_final_model or save_best_model: # Get a model id
        
        dest_model = cgf['MODEL_METADATA']['dest_model']
        model_number_type = cgf['MODEL_METADATA']['model_number_type']
        if model_number_type.lower() == 'fixed':
            model_number = cgf['MODEL_METADATA']['model_number_arguments']['model_id']
        elif model_number_type.lower() == 'count':
            counter_path =\
                cgf['MODEL_METADATA']['model_number_arguments']['counter_path']
            model_number = read_count(counter_path);
        elif model_number_type.lower() == 'uuid':
            model_number = uuid.uuid1().int
        else:
            model_number = -1;
        
        full_dest_model = join(dest_model, str(model_number))
        if not os.path.isdir(dest_model):
            os.mkdir(dest_model)
        if not os.path.isdir(full_dest_model):
            os.mkdir(full_dest_model)
        else:
            print('Delete all content in the folder: \n{}'.format(full_dest_model))
            shutil.rmtree(full_dest_model);
            os.mkdir(full_dest_model)
        shutil.copyfile('config.yml', join(full_dest_model, 'config.yml'))
        shutil.copyfile('model_builders.py', join(full_dest_model, 'model_builders.py'))
        print("""Model nbr type: {}
Model number: {}
Model dest: {}""".format(model_number_type, model_number, dest_model))
   
    print('Save best model: {}'.format(save_best_model))
    
    ''' 
    if save_best_model:
        arguments = cgf['MODEL_METADATA']['save_best_model']['arguments']
        for key, value in arguments.items():
            print('\t{}: {}'.format(key,value))

        filename = arguments['filepath']
        arguments['filepath'] = join(full_dest_model, filename)
        mc = ModelCheckpoint(arguments['filepath'], monitor=arguments['monitor'], verbose=arguments['verbose'], save_best_only=False, save_weights_only=True, mode=arguments['mode'], period=arguments['period'])
        if callbacks is None:
            callbacks = mc
        elif type(callbacks) is list: # Is a list
            callbacks.append(mc)
        else:
            callbacks = mc
            print('ERROR: Unknown callback type')
    '''
    '''
    # We save weights after each epoch
    if callbacks == None:
        callbacks =[]
    mycallback = MyCallback()
    callbacks.append(mycallback)
    '''

    print('\nStart training the model\n')
    # Train model :)
    print(shuffle_data)
    print(callbacks)
    

    data_loader_test = data_selector(cgf['DATASET_TEST']['name'], cgf['DATASET_TEST']['arguments'])
    test_data, test_labels = data_loader_test.load_data()
    optimal = 0

    while optimal == 0 :
        model = model_selector(model_name, input_shape, output_shape, model_arguments)
        model.compile(optimizer = optimizer,
                      loss = loss_type,
                      metrics = metric_list)
        history = model.fit(train_data, train_labels, 
                  epochs=max_epoch, 
                  batch_size=batch_size,
                  validation_data=(val_data, val_labels),
                  shuffle=shuffle_data,
                  callbacks = callbacks)

        score = model.evaluate(test_data, test_labels, verbose=0)
        if score[1] == 0:
            optimal = 1

    if save_final_model:
        full_file_name = join(full_dest_model, 'keras_model_files.h5')
        model.save(join(full_dest_model, 'keras_model_files.h5'), save_format='h5')
        print('Saved model as {}'.format(full_file_name))

    # plot the loss history
    plt.figure()
    loss_hist = history.history['loss'][int(max_epoch/2):]
    index = np.linspace(max_epoch/2+1, max_epoch, max_epoch/2);
    plt.plot(index,loss_hist);
    plt.savefig(join(full_dest_model, 'loss_graph.png'));
    # Load and test model on test set
    
    print('\nDATASET TEST')
    print(data_loader_test)

    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Accuracy on test set: {}%'.format(100*score[1]))

