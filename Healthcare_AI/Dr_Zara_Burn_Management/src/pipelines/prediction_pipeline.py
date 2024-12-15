import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        try:
            # Load the trained model
            logging.info("Loading model...")
            model = load_model(os.path.join("model", "model.h5"))
            logging.info("Model loaded successfully.")

            # Preprocess the image
            imagename = self.filename
            test_image = image.load_img(imagename, target_size=(299, 299))
            test_image = image.img_to_array(test_image)
            test_image = test_image / 255.0  # Normalize the image
            test_image = np.expand_dims(test_image, axis=0)

            # Make a prediction
            logging.info("Making prediction...")
            result = np.argmax(model.predict(test_image), axis=1)
            print(result)

            if result[0] == 0:
                prediction = 'First_Degree_Burn'
                return prediction
            elif result[0] == 1:
                prediction = 'Second_Degree_Burn'
                return prediction
            else:
                prediction = 'Third_Degree_Burn'
                return prediction

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e
