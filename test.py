import tensorflow as tf
import keras.datasets.mnist as ms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#Give to me the fashion mnist dataset
def clothes():
    (x_train, y_train), (x_test, y_test) = ms.load_data()
    #Normalize the data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    #make class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #Create the model
    #Load model
    model = tf.keras.models.load_model('epic_num_reader.model.h5')

    def make_img():
        #I will get the path of the haine.py file
        path = os.path.dirname(os.path.realpath(__file__))
        #I will get the path of the image
        imgpath = os.path.join(path, '\test.jpg')
        print(path)

        #I will load the image
        img = cv2.imread(imgpath)
        #I will convert the image to grayscale
        if img is not None:
            # do image processing here
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("Image not loaded successfully")
        #I will show the image
        plt.imshow(gray, cmap='gray')
        #I will resize the image
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        #I will show the image
        plt.imshow(resized, cmap='gray')
        #I will normalize the image
        newimg = tf.keras.utils.normalize(resized, axis=1)
        #I will show the image
        plt.imshow(newimg, cmap='gray')
        #I will make a prediction
        predictions = model.predict(np.array([newimg]))
        #I will print the prediction
        print(np.argmax(predictions))


    # program to capture single image from webcam in python

    # importing OpenCV library
    from cv2 import *

    # initialize the camera
    # If you have multiple camera connected with
    # current device, assign a value in cam_port
    # variable according to that
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    # reading the input using the camera
    result, img = cam.read()

    # If image will detected without any error,
    # show result
    if result:

        # showing result, it take frame name and image
        # output
        if img is not None:
            # do image processing here
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("Image not loaded successfully")
        #I will show the image
        plt.imshow(gray, cmap='gray')
        #I will resize the image
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        #I will show the image
        plt.imshow(resized, cmap='gray')
        #I will normalize the image
        newimg = tf.keras.utils.normalize(resized, axis=1)
        #I will show the image
        plt.imshow(newimg, cmap='gray')
        #I will make a prediction
        predictions = model.predict(np.array([newimg]))
        plt.show()
        #I will print the prediction
        print(class_names[np.argmax(predictions)])
        return class_names[np.argmax(predictions)]

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")