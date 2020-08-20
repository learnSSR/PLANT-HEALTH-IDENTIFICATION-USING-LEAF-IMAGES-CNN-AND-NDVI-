import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras

classifier = keras.models.load_model('sample_data/model.h5')

def plot_image(filepath):
    test_img=cv2.imread(filepath,1)
    plt.figure()
    plt.subplot(1,1,1)
    plt.imshow(test_img)
    plt.title('Test Image')
    plt.show()

def ndvi_prediction(filepath):
    img = cv2.imread(filepath, 1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for x in range(0,256):
        for y in range(0,256):
            if img_hsv[x][y][1] < 51:
                img[x][y][0]=0
                img[x][y][1]=0
                img[x][y][2]=0
    
    unhealthy = 0
    actual_pixels = 0
    for x in range(0,256):
        for y in range(0,256):
            if img[x][y][0] != 0 and img[x][y][1] != 0 and img[x][y][2] != 0:
                actual_pixels += 1
                nir = ((1.8474519*img[x][y][2])-(0.1936929*img[x][y][1])+(0.12401134*img[x][y][0]))
                if (nir - img[x][y][2]) / (nir + img[x][y][2]) < 0.2463804:
                    unhealthy += 1

    if unhealthy / actual_pixels >= 0.19:
        return "unhealthy"
    else:
        return "healthy"

def cnn_prediction(filepath, classifier):
    img = keras.preprocessing.image.load_img(
        filepath, target_size=(64, 64)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = classifier.predict(img_array)
    predicted_class_index=pred.argmax(axis=-1)
    labels = ["healthy", "unhealthy"]
    cnn_output = [labels[k] for k in predicted_class_index][0]
    return cnn_output

def compare_output(cnn_output, ndvi_output):
        if ndvi_output == "unhealthy" and cnn_output == "healthy":
            return "unhealthy"
        else:
            return cnn_output
