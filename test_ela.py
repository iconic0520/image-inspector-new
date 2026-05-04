import os
import numpy as np
from keras.models import load_model
from helper import prepare_image_for_ela

def test_detection():
    model_path = 'ELA_Training/model_ela.h5'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Use one of the existing images for testing
    test_img = 'imgs/org1.jpg'
    if not os.path.exists(test_img):
        print(f"Test image not found at {test_img}")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Processing image {test_img}...")
    np_img_input, ela_img = prepare_image_for_ela(test_img)
    
    print("Running prediction...")
    Y_predicted = model.predict(np_img_input, verbose=0)
    
    class_ELA = ['Real', 'Tampered']
    accuracy = round(np.max(Y_predicted[0]) * 100)
    prediction = class_ELA[np.argmax(Y_predicted[0])]
    
    print(f"Result: {accuracy}% confidence of image being {prediction}")
    print("Test passed!")

if __name__ == "__main__":
    test_detection()
