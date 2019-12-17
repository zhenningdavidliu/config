import tensorflow as tf

from tensorflow.keras.datasets import mnist, cifar10, cifar100

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

import numpy as np
import random 
import os
import matplotlib.pyplot as plt
import foolbox


def create_model(img_rows, img_cols, channels):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

def adversarial_pattern(image,label):
    
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = tf.keras.losses.MSE(label,prediction)

    gradient = tape.gradient(loss,image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad

if __name__ == '__main__':

# Set up computational resource
    use_gpu = True 
    print("""\nCOMPUTER SETUP
Use gpu: {}""".format(use_gpu))
    if use_gpu:
        compute_node = 3
        os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
        print('Compute node: {}'.format(compute_node))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

    (x_train_raw, y_train_raw), (x_test, y_test) = mnist.load_data()

    labels = ['zero','one','two','three','four','five','six','seven','eight','nine']

    img_rows, img_cols, channels = 28, 28, 1
    num_classes = 10

    x_train = x_train_raw / 255
    x_test = x_test / 255

    x_train = x_train.reshape((-1, img_rows, img_cols, channels))
    x_test = x_test.reshape((-1, img_rows, img_cols, channels))
    y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    kmodel = create_model(img_rows, img_cols, channels)  

    kmodel.fit(x_train, y_train,
              batch_size=32,
              epochs=1,
              validation_data=(x_test, y_test))
    model = foolbox.models.TensorFlowEagerModel(kmodel,bounds=(0,1))
    print("Data shapes", x_test.shape, y_test.shape, x_train.shape, y_train_raw.shape)

    x_train_0 = np.transpose(x_train, (0,3,1,2))
    print(x_train_0.shape)
    print(np.mean(model.forward(x_train).argmax(axis=-1)==y_train))
    attack = foolbox.v1.attacks.DeepFoolAttack(model)
    adversarials = attack(x_train, y_train_raw)

    print(np.argmax(model.forward_one(adversarial)))
    preds = kmodel.evaluate(adversarials[np.newaxis].copy(), y_train, verbose=0)
    print(type(adversarials))
    print(preds)
'''
peturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label).numpy()

adversarial = image + peturbations*0.1

if channels == 1:
    plt.imshow(adversarial.reshape((img_rows,img_cols)))
else:
    plt.imshow(adversarial.reshape((img_rows,img_cols,channels)))
plt.show()
'''
