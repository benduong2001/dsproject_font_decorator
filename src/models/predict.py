#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from tensorflow import keras
import os
import numpy as np
from tqdm import tqdm

class CNN_Model_Prediction:
    def __init__(self, path_file_model="../src/models/saved/cnn_model.h5"):
        #path_file_model = os.path.join(os.getcwd(), "saved", "cnn_model.h5")
        #print(path_file_model)
        self.set_model(path_file_model)
    def set_model(self, path_file_model):
        self.model = keras.models.load_model(path_file_model)
        #print(self.model)
    def image_preprocessing(self, image):
        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = gray
        img = 255-img

        cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
        return img
    def get_contours(self, image):
        img = image

        find_contours_output = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        contours = find_contours_output[0]
        hier =  find_contours_output[1]
        return contours
    def predict_contour(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        max_x = np.max(box[:,0])
        min_x = np.min(box[:,0])
        max_y = np.max(box[:,1])
        min_y = np.min(box[:,1])

        extraction = self.image[min_y:max_y,min_x:max_x]
        #print("IMAGE SHAPE", extraction.shape)
        if np.prod(np.array(list(extraction.shape))) > 0:
            prediction = self.model.predict(np.array([extraction]))[0]
        else:
            prediction = None
        return prediction
    def predict_contours(self, contours):
        predictions = []
        for contour_i in tqdm(range(len(contours))):
            contour = contours[contour_i]
            prediction = self.predict_contour(contour)
            if ((prediction is None)==False):
                predictions.append(prediction)
        if predictions == []:
            result = [0,0,0]
        else:
            imputed_predictions = (np.nan_to_num(np.array(predictions)))
            
            #predictions_avg = (np.nan_to_num(np.array(predictions)).mean(axis=0))
            #result = list(predictions_avg)
            imputed_predictions_column_wise_sums = imputed_predictions.sum(axis=0)
            result = imputed_predictions_column_wise_sums / np.sum(imputed_predictions_column_wise_sums)

        return result
    def predict(self, image):
        image = self.image_preprocessing(image)
        contours = self.get_contours(image)
        self.image = image
        result = self.predict_contours(contours)
        return result

