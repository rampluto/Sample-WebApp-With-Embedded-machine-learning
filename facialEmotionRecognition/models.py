import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

class Utility:
    def __init__(self):
        pass

    def get_model(self):
        global model
        model = keras.models.load_model('emotionRecog.h5')
        print("Model loaded!")

    def load_image(self,img_path):
        img = image.load_img(img_path, target_size=(16, 48))
        img_tensor = image.img_to_array(img)            
        img_tensor = np.expand_dims(img_tensor, axis=0) 
        return img_tensor

    def prediction(self,img_path):
        emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        tests = self.load_image(img_path)
        tests = np.array(tests).reshape(-1)
        tests = tests.reshape(1,48,48,1)
        y = model.predict(tests)
        classe = np.argmax(y)
        return emotions[classe]


