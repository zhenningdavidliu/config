import tensorflow as tf
import os
from os.path import join

class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
            # things done on end of the epoch
            if epoch % 100 == 0: 
                self.model.save_weights("model/7/weights_at_epoch-%d.h5" %epoch) # save the model
