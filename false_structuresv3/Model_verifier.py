import yaml
import numpy as np
import tensorflow  as tf
import model_builders as mb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from Data_loader_stripe import Data_loader_stripe_test 

def load_test(data_name, arguments):
    
    if data_name.lower() == "stripe_test":
       return Data_loader_stripe_test(arguments) 
    else:
        print('Error: Could not find data loader with name {}' .format(data_name))       
        return None

if __name__ == "__main__":
    """Load trained model and check whether it generalized to a larger Test set"""
    

    #Load configuration file
    with open('verify_config.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);        

    #Load model
    weights_path = cgf['MODEL']['filepath']
    n = cgf['MODEL']['arguments']['grid_size']
    input_shape=(n,n,1)
    model = mb.build_model_fc3(input_shape, cgf['MODEL']['arguments']['output_shape'], cgf['MODEL']['arguments'])
    model.load_weights(weights_path)

    optimizer=cgf['MODEL']['compilation']['optim']['type']
    loss_type=cgf['MODEL']['compilation']['loss']['type']
    metrics_list=list(cgf['MODEL']['compilation']['metric'].values())


    model.compile(optimizer = optimizer,
                 loss = loss_type,
                 metrics = metrics_list)

    #Test model 
    data_loader_test =  load_test(cgf['DATASET_TEST']['name'],cgf['DATASET_TEST']['arguments'])

    test_data, test_labels = data_loader_test.load_data()
    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Accuracy on the test set: {}%' .format(100*score[1]))
