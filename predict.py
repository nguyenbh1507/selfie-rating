# Import statement - popular use
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

from keras.models import load_model

def predict(model, img):
    
    """
    Calculate prediction score of `model` over a `img` file
    """
    imgA = cv2.imread(img)
    imgA = cv2.resize(imgA, (224, 224))/255.0
    return model.predict(imgA.reshape(1,224,224,3))

# load model
model = load_model('beauty_model.h5')

test_folder = './Selfie-dataset/images'
start = time.time()
test_results = []
image_files = os.listdir(test_folder)

print('*'*50, '\nScoring images', '.'*20, end='')
i = 1
for file in os.listdir(test_folder):
    # for loop is slow but it keeps the order which file corresponding to each score
    path = os.path.join(test_folder, file)
    test_results.append((file, predict(model, path)[0][0]))
    if i%1000 == 0:
        print('Processed %d/%d images' % (i, len(image_files)))
    i += 1
print("Predicted in {} seconds".format(int(time.time()-start)))

# Get top 5
top5 = sorted(test_results, key=lambda x: x[1], reverse=True)[:5]
# show the results
i = 1
plt.figure(figsize=(15,10))
for img, score in top5:
    plt.subplot(2,3,i)
    plt.title('File: {}, \nScore: {}'.format(img, score), fontsize=10)
    plt.imshow(plt.imread(os.path.join(test_folder, img)))
    i += 1

plt.show()