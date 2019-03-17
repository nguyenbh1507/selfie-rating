# Import statement - popular use
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from functions import *


# random seed, this helps to reproduce Keras results
np.random.seed(1000) # NumPy
import random
random.seed(1000) # Python
from tensorflow import set_random_seed
set_random_seed(1000) # Tensorflow

# load dataset
image_folder = './SCUT-FBP5500_v2/Images'
scoring_file = './SCUT-FBP5500_v2/train_test_files/All_labels.txt'

file_name_list, file_name_indices, scores = load_file_list(scoring_file)
print('*'*50, '\nLoading images ...')
images = load_images(image_folder, file_name_list)

# load train/test image indices
train_img_file = './SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt'
test_img_file = './SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/test.txt'
train_indices = filelist2indices(train_img_file, file_name_indices)
test_indices = filelist2indices(test_img_file, file_name_indices)

# resize images for test
print('*'*50, '\nResizing and normalization ...')
images_R = []
for i in range(images.shape[0]):
    images_R.append(cv2.resize(images[i], (224,224)))
images_R = np.array(images_R)
# normalization
# 16GB memory will allocated because of float type
images_R = images_R/255.0

# define train, test data
X_train = images_R[train_indices]
X_test = images_R[test_indices]
y_train = scores[train_indices]
y_test = scores[test_indices]

model = create_model()
print(model.summary())
# Save model architecture
with open('beauty_model_architecture.json', 'w') as f:
    f.write(model.to_json())

# train model
print('*'*50, 'Training model ...')
his = model.fit(x=X_train, y=y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))

# save model
model.save('beauty_model.h5')

# plot train & test loss
plt.plot(his.history['loss'], label='Train MSE')
plt.plot(his.history['val_loss'], label='Test MSE')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print('Done!\n', '='*50)