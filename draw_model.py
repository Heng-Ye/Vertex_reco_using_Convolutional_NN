import sys
from keras.utils import plot_model

import argparse
parser = argparse.ArgumentParser(description='Run CNN training on patches with a few different hyperparameter sets.')
parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
parser.add_argument('-o', '--output', help="Output model file name", default='model')
parser.add_argument('-g', '--gpu', help="Which GPU index", default='0')
args = parser.parse_args()

import os
os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import tensorflow as tf
import keras
if keras.__version__[0] != '2':
    print 'Please use the newest Keras 2.x.x API with the Tensorflow backend'
    quit()
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')

import numpy as np
np.random.seed(2017)  # for reproducibility
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from os.path import exists, isfile, join
import json

from utils import read_config, get_patch_size, count_events


'''
def save_model(model, name):
    try:
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True   # Save successful
    except:
        return False  # Save failed


#######################  configuration  #############################
print 'Reading configuration...'
config = read_config(args.config)

CNN_INPUT_DIR = config['training_on_patches']['input_dir']
# input image dimensions
PATCH_SIZE_W, PATCH_SIZE_D = get_patch_size(CNN_INPUT_DIR)
img_rows, img_cols = PATCH_SIZE_W, PATCH_SIZE_D

batch_size = config['training_on_patches']['batch_size']
nb_classes = config['training_on_patches']['nb_classes']
nb_epoch = config['training_on_patches']['nb_epoch']

nb_pool = 2 # size of pooling area for max pooling

cfg_name = 'sgd_lorate'

# convolutional layers:
nb_filters1 = 48  # number of convolutional filters in the first layer
nb_conv1 = 5      # 1st convolution kernel size
convactfn1 = 'relu'

maxpool = False   # max pooling between conv. layers

nb_filters2 = 0   # number of convolutional filters in the second layer
nb_conv2 = 7      # convolution kernel size
convactfn2 = 'relu'

drop1 = 0.2

# dense layers:
densesize1 = 128
actfn1 = 'relu'
densesize2 = 32
actfn2 = 'relu' 
drop2 = 0.2

#######################  CNN definition  ############################
print 'Compiling CNN model...'
with tf.device('/gpu:' + args.gpu):
    main_input = Input(shape=(img_rows, img_cols, 1), name='main_input')

    if convactfn1 == 'leaky':
        x = Conv2D(nb_filters1, (nb_conv1, nb_conv1),
                   padding='valid', data_format='channels_last',
                   activation=LeakyReLU())(main_input)
    else:
        x = Conv2D(nb_filters1, (nb_conv1, nb_conv1),
                   padding='valid', data_format='channels_last',
                   activation=convactfn1)(main_input)

    if nb_filters2 > 0:
        if maxpool:
	    x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)
        x = Conv2D(nb_filters2, (nb_conv2, nb_conv2))(x)
        if convactfn2 == 'leaky':
            x = Conv2D(nb_filters2, (nb_conv2, nb_conv2), activation=LeakyReLU())(x)
        else:
            x = Conv2D(nb_filters2, (nb_conv2, nb_conv2), activation=convactfn2)(x)

    x = Dropout(drop1)(x)
    x = Flatten()(x)
    # x = BatchNormalization()(x)

    # dense layers
    x = Dense(densesize1, activation=actfn1)(x)
    x = Dropout(drop2)(x)

    if densesize2 > 0:
        x = Dense(densesize2, activation=actfn2)(x)
        x = Dropout(drop2)(x)

    # outputs
    el_inel_none = Dense(3, activation='softmax', name='el_inel_none_netout')(x)

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model = Model(inputs=[main_input], outputs=[el_inel_none])
    #model.compile(optimizer=sgd,
                  #loss={'el_inel_none_netout': 'categorical_crossentropy'})



    

plot_model(model, to_file='model.png')


'''
