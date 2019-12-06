import yaml 
import numpy as np
import tensorflow
import model_builders as mb
from Data_loader_stripe import Data_loader_stripe_test
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten

def random_labeled_data(data_size,randomness):
    """	
    We load a trained model and generate a train set either using 
    uniform or a gaussian random matrices
    """

    with open('config.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);
    n = cgf['DATASET_TRAIN']['arguments']['grid_size']
    input_shape = (n,n,1)
    output_shape = (1)	

    weights_path = "model/7/keras_model_files.h5"	
    model = mb.build_model_fc2(input_shape, output_shape,cgf['MODEL']['arguments'])
    model.load_weights(weights_path)

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type= cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    model.compile(optimizer=optimizer,
                 loss=loss_type,
                 metrics= metric_list)
    
    if randomness == "gaussian":
        #Gaussian with total mean 96 
        data = np.random.normal(0.09375, size=(data_size, n, n, 1))
    elif randomness == "uniform":				
        #data = np.random.uniform(low=0, high=0.1875, size=(data_size, n, n, 1)) 
        data = np.random.uniform(low=-0.5, high=2, size=(data_size, n, n, 1)) 

    elif randomness == "stripes":
        data_loader_test = Data_loader_stripe_test(cgf['DATASET_TEST']['arguments'])
        data, _ =  data_loader_test.load_data() 

    sum_pixels = [i.sum() for i in data[:]]

    labels = model.predict(data)
    return data, labels, sum_pixels
