import os

import cv2
import numpy as np
import tensorflow as tf

class ImageRecogniser():
    def __init__(self, model_dir, image_pixels):
        self.image_pixels = image_pixels
        self.learn_rate = 1e-3
        # sizes must match
        self.model_name = 'catdetectv3-{}-{}.model'.format(self.learn_rate,'5conv-basic')
        self.model_dir = model_dir
        self.debugTimer = DebugTimer.DebugTimer(["ConvertImage", "RecogniseImage"])

    def start(self):
        return self.create_graph()

    def create_graph(self):
        # Create graph
        from tflearn.layers.conv import conv_2d, max_pool_2d
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.estimator import regression
        import tflearn

        convnet = input_data(shape=[None, self.image_pixels, self.image_pixels, 3], name='input')

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
        convnet = regression(convnet, optimizer='adam', learning_rate=self.learn_rate, loss='categorical_crossentropy',
                             name='targets')

        self.model = tflearn.DNN(convnet, tensorboard_dir='log')

        model_fileName = os.path.join(self.model_dir, '{}'.format(self.model_name))
        print("Attempting to load model file", model_fileName)
        if os.path.exists(model_fileName + ".meta"):
            self.model.load(model_fileName)
            print('Model loaded!')
            return True
        else:
            print("Model file doesn't exist")
        return False

    def recogniseImage(self, sess, image, num_top_predictions):

        self.debugTimer.start(0)
        # Convert image
        img2 = cv2.resize(image, dsize=(self.image_pixels, self.image_pixels), interpolation=cv2.INTER_CUBIC)
        # Numpy array
        np_image_data = np.array(img2)
        # np_image_data = np.asarray(img2)
        # np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        # maybe insert float convertion here - see edit remark!
        # np_final = np.expand_dims(np_image_data, axis=0)
        data = np_image_data.reshape(self.image_pixels, self.image_pixels, 3)
        self.debugTimer.end(0)

        self.debugTimer.start(1)
        model_out = self.model.predict([data])[0]

        if np.argmax(model_out) == 1:
            str_label = 'good'
        else:
            str_label = 'bad'

        self.debugTimer.end(1)
        return (str_label, 100)

if __name__ == '__main__':
    import DebugTimer
    with tf.Session() as sess:
        testDirName = "../../nnTestImages"
        testImgSize = 50
        imgRec = ImageRecogniser("", testImgSize)
        imgRec.start()
        for imgName in os.listdir(testDirName):
            (fname, ext) = os.path.splitext(imgName)
            if ext != ".jpg":
                continue
            path = os.path.join(testDirName, imgName)
            img = cv2.imread(path)
            str_label, perc = imgRec.recogniseImage(sess, img, 1)
            print(imgName, str_label)
        imgRec.debugTimer.printTimings()
else:
    from Utils import DebugTimer

