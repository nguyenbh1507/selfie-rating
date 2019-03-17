# Import statement - popular use
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Model import statement
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam

os.environ['KERAS_BACKEND'] = 'tensorflow'


def load_file_list(scoring_file):
    """
    Load filenames, their score and indices
    `scoring_file`: path to file of all scores
    return:
        - filenames -- list, control the order
        - dictionary -- {filename: index}
        - scores -- numpy array
    """
    
    file_name_list = []
    file_name_indices = {}
    scores = []
    
    with open(scoring_file, 'rt') as imgnames:
        lines = imgnames.readlines()
        for i in range(len(lines)):
            name, score = lines[i].strip().split(' ')
            file_name_indices[name] = i
            file_name_list.append(name)
            scores.append(float(score))
        scores = np.array(scores).reshape(-1, 1)
    return file_name_list, file_name_indices, scores


def load_images(image_folder, file_name_list):
    """
    Load images in image_folder to a numpy array, shape = [#files, width, height, chanel]
    Filnames are listed in `file_name_list`
    `image_folder` should be include all files in `file_name_list`, otherwise error will raise
    The results will include a numpy array, so this method should be used with small dataset
    Return
        - numpy array of images
    """
    img_data = []
    for i in range(len(file_name_list)):
        img_name = file_name_list[i]
        img = plt.imread(os.path.join(image_folder, img_name))
        img_data.append(img)
    
    return np.array(img_data) 


def filelist2indices(filelist, inddict):
    """
    Return indices of files from filelist in inddict
    This indices will be used to extract train/test from full loaded dataset
    `filelist`: file including line = [filename score]
    `inddict`: {filename: index} 
    """
    idx = []
    with open(filelist, 'rt') as file:
        for line in file.readlines():
            img_name = line.strip().split(' ')[0]
            idx.append(inddict[img_name])
    return np.array(idx)


def create_model():
    """
    Create model
    """
    
    # model creation
    # similar to AlexNet architecture

    model = Sequential()
    # # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), 
                     padding='valid', activation='relu', kernel_initializer='he_normal', name='CONV_1'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MAX_POOL_1'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), 
                     padding='valid', activation='relu', kernel_initializer='he_normal', name='CONV_2'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MAX_POOL_2'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
                     padding='valid', activation='relu', kernel_initializer='he_normal', name='CONV_3'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
                     padding='valid', activation='relu', kernel_initializer='he_normal', name='CONV_4'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), 
                     padding='valid', activation='relu', kernel_initializer='he_normal', name='CONV_5'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', name='MAX_POOL_5'))

    # ADD flatten
    model.add(Flatten(name="FLATTEN"))
    # ADD dense layer 512 hidden unit
    model.add(Dense(units=32, activation='relu', name="Dense_1"))
    # ADD dense layer
    model.add(Dense( units=1, name="Dense_2", activation='relu'))

    # optimizer
    optimizer = Adam(lr=1e-4)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model