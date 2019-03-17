# Import statement - popular use
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from functions import *

from keras.models import model_from_json, load_model

# load dataset
image_folder = './SCUT-FBP5500_v2/Images'
scoring_file = './SCUT-FBP5500_v2/train_test_files/All_labels.txt'

file_name_list, file_name_indices, scores = load_file_list(scoring_file)
print('*'*50, '\nLoading images ...')
images = load_images(image_folder, file_name_list)

# resize images for test
print('*'*50, '\nResizing and normalization ...')
images_R = []
for i in range(images.shape[0]):
    images_R.append(cv2.resize(images[i], (224,224)))
images_R = np.array(images_R)
# normalization
# 16GB memory will allocated because of float type
images_R = images_R/255.0

print('Begin cross validation on 5 folds ...')
cross_folder = './SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files'
his = []
for i in range(1,6):
    print('-'*50,'\nCross validation for %d-th splitting'%i)
    # Model load from h5 file, it's better to continue training from a quite good model
    model = load_model('beauty_model.h5')   
    
    # load train/test image indices
    train_img_file = os.path.join(cross_folder, 'cross_validation_%d'%i, 'train_%d.txt'%i)
    test_img_file = os.path.join(cross_folder, 'cross_validation_%d'%i, 'test_%d.txt'%i)
    
    # load indices
    train_indices = filelist2indices(train_img_file, file_name_indices)
    test_indices = filelist2indices(test_img_file, file_name_indices)
    
    # split to train/test
    X_train = images_R[train_indices]
    X_test = images_R[test_indices]
    y_train = scores[train_indices]
    y_test = scores[test_indices]
    
    # fit the model
    h = model.fit(x=X_train, y=y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
    his.append(h)

print('RMSE:\n')
print([np.sqrt(h.history['val_loss'][-1]) for h in his])
print('Done!\n', '='*50)  
    