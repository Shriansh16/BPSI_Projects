import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self, model):
        try:
            # Preprocess the image
            logging.info(f"Processing image: {self.filename}")
            test_image = image.load_img(self.filename, target_size=(299, 299))
            test_image = image.img_to_array(test_image)
            test_image = test_image / 255.0  # Normalize the image
            test_image = np.expand_dims(test_image, axis=0)

            # Make a prediction
            logging.info("Making prediction...")
            result = np.argmax(model.predict(test_image), axis=1)

            # Prediction categories for burn degrees
            if result[0] == 0:
                prediction = 'First_Degree_Burn'
            elif result[0] == 1:
                prediction = 'Second_Degree_Burn'
            else:
                prediction = 'Third_Degree_Burn'
                
            return prediction, result[0]

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e