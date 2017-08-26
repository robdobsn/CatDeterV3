import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import time

TEST_DIR = '../../nnTestImages'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'catdetectv3-{}-{}.model'.format(LR, '5conv-basic') # just so we remember which saved model is which, sizes must match

def process_test_data():
    testing_data = []
    for imgName in tqdm(os.listdir(TEST_DIR)):
        (fname, ext) = os.path.splitext(imgName)
        if ext != ".jpg":
            continue
        path = os.path.join(TEST_DIR, imgName)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), imgName])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('../ClassifyBadCats/{}.meta'.format(MODEL_NAME)):
    print("Loading model ... ", end="")
    model.load(MODEL_NAME)
    print('model loaded!')

fig = plt.figure()

totalTime = 0
totalImages = 0
totalCorrect = 0
totalFalsePositives = 0
for num, data in enumerate(test_data):
    # bad: [1,0]
    # good: [0,1]

    img_name = data[1]
    img_data = data[0]

    word_label = img_name.split('_')[0]

    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
    # model_out = model.predict([data])[0]
    startTime = time.time()
    model_out = model.predict([data])[0]
    endTime = time.time()
    totalTime += endTime-startTime
    totalImages += 1

    if np.argmax(model_out) == 1:
        str_label = 'Good'
        if word_label == "good":
            totalCorrect += 1
    else:
        str_label = 'Bad'
        if word_label == "bad":
            totalCorrect += 1
        else:
            totalFalsePositives += 1

    if num < 32:
        y = fig.add_subplot(4, 8, num + 1)
        orig = cv2.imread(os.path.join(TEST_DIR, img_name))
        y.imshow(orig)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

print("Correct {:0.2f}% Mean time {:0.1f}ms (Total time {:0.3f}s, count {}, false+ve {})".format(100*totalCorrect/totalImages, 1000*totalTime/totalImages, totalTime, totalImages, totalFalsePositives))
plt.show()

