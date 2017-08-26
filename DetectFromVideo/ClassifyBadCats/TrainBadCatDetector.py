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

# Change this to True to force recreation of the training data from the input images
CREATE_TRAIN_DATA = False

TRAIN_DIR = '../../trainImages'
IMG_SIZE = 50
LR = 1e-3
TRAIN_DATA_FILE_NAME = 'train_data.npy'

MODEL_NAME = 'catdetectv3-{}-{}.model'.format(LR, '5conv-basic') # just so we remember which saved model is which, sizes must match

def label_img(img):
    word_label = img.split('_')[0]
    # conversion to one-hot array [bad,good]
    if word_label == 'bad': return [1,0]
    elif word_label == 'good': return [0,1]
    print("Incorrect label")
    exit(0)

def create_train_data():
    training_data = []
    for imgName in tqdm(os.listdir(TRAIN_DIR)):
        (fname, ext) = os.path.splitext(imgName)
        if ext != ".jpg":
            continue
        label = label_img(imgName)
        path = os.path.join(TRAIN_DIR,imgName)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save(TRAIN_DATA_FILE_NAME, training_data)
    return training_data

if CREATE_TRAIN_DATA or not os.path.exists(TRAIN_DATA_FILE_NAME):
    train_data = create_train_data()
else:
    # If you have already created the dataset:
    train_data = np.load(TRAIN_DATA_FILE_NAME)

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

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

