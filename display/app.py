import os
print(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, render_template
import pandas as pd
#import pickle
#import joblib
import numpy as np
import cv2
import urllib

import sys
sys.path.append("..")
from src.models.predict import *



# Declare a Flask app
app = Flask(__name__)

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        ## # Unpickle classifier
        # model = joblib.load("../data/final/dtr.pkl")
        
        image_url = request.form.get("image_url")

        try:
            url_response = urllib.request.urlopen(image_url)
            image = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8),-1)
            image = image[:,:,::-1]

            cnn_model_prediction = CNN_Model_Prediction()
            prediction_output = cnn_model_prediction.predict(image)
            try:
                html_output_value = str(prediction_output)
            except:
                html_output_value = "Error"
        except:
            html_output_value = "Forbidden URL"
    else:
        html_output_value = ""
        
    return render_template("website.html", output = html_output_value)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
