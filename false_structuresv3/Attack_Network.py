import tensorflow as tf
import foolbox
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Conv1D, Conv2D, Flatten
from Data_loader_stripe import Data_loader_stripe_test
import os
import yaml
import model_builders as mb
from os.path import join

def generate_perturbed_image(modelweights_id, imageset, imagelabels, size):
    with open('config.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);
    n = cgf['DATASET_TRAIN']['arguments']['grid_size']
    input_shape = (n,n,1)
    output_shape = (1)

    weights_path = join("model/",str(modelweights_id),"keras_model_files.h5")
    print(weights_path)

    model = mb.build_model_fc3(input_shape, output_shape, cgf['MODEL']['arguments'])
    model.load_weights(weights_path)

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type = cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    model.compile(optimizer = optimizer,
                  loss = loss_type,
                  metrics = metric_list)

    kmodel = foolbox.models.TensorFlowEagerModel(model,bounds=(-0.01,1.01))
    attack = foolbox.v1.attacks.DeepFoolAttack(kmodel)
    adversarials = [attack(imageset[i],np.float32(np.asscalar(np.array(imagelabels[i])))) for i in range(size)]

    return adversarials


