from Attack_Network import generate_perturbed_image
import numpy as np
import tensorflow as tf
from Data_loader_stripe import Data_loader_stripe_train
import yaml

if __name__ == "__main__":
    with open('config.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    data_loader_train = Data_loader_stripe_train(cgf['DATASET_TRAIN']['arguments'])
    perturbed_images=np.array(0)
    images, labels = data_loader_train.load_data()
    for i in range(6,7,1):
        np.concatenate((perturbed_images,generate_perturbed_image(i+1,images, labels, np.shape(images)[0])), axis=0)
