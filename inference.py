import time
import json
import requests

import numpy as np
import tensorflow as tf

IMAGE_SIZE = 224  # Define IMAGE_SIZE if it's not defined elsewhere
MODEL_URL = 'http://localhost:8501/v1/models/pets:predict'
CLASSES = ['Cat', 'Dog']


def get_prediction(image_path):
    start_time = time.time()
    prediction_score = -1.0
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = tf.keras.applications.mobilenet_v2.preprocess_input(input_arr)
        input_arr = np.expand_dims(input_arr, axis=0)

        data = json.dumps({"signature_name": "serving_default", "instances": input_arr.tolist()})
        response = requests.post(MODEL_URL, data=data.encode())

        # Print detailed information about the response
        print("Status Code:", response.status_code)  # Status Code: 200
        print("Headers:", response.headers)  # Headers: {'Content-Type': 'application/json', 'Date': 'Fri, 09 Aug 2024 13:50:49 GMT', 'Content-Length': '41'}
        print("Response Text:", response.text)
        # Response Text: {
        #     "predictions": [[0.0859318]]
        # }

        print("Response JSON:", response.json())  # Response JSON: {'predictions': [[0.0859318]]}
        result = json.loads(response.text)
        predictions = result.get('predictions', [])
        if predictions:
            prediction_score = predictions[0][0]
            print("Prediction Score:", prediction_score)  # Prediction Score: 0.0859318
        else:
            print("No predictions found in the response.")
    except ImportError as e:
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    class_name = CLASSES[int(prediction_score >= 0.5)]
    print(f"Finished in duration {time.time()-start_time}(s)")
    return class_name

# Test the function with an image path
# image_path = './corpus/cat/c1.jpg'
# get_prediction(image_path=image_path)
# (py310_flask) > python .\inference.py
# [[0.0859318]]
