import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from helper import prepare_image_for_ela
from PIL import Image

def analyze_image(img_path, model_path):
    print(f"Analyzing image: {img_path}")
    print(f"Using model: {model_path}")
    
    if not os.path.exists(img_path):
        print("Error: Image not found.")
        return
    
    if not os.path.exists(model_path):
        print("Error: Model not found.")
        return

    try:
        # Prepare the image
        np_img_input, ela_img = prepare_image_for_ela(img_path)
        
        # Load the model
        model = load_model(model_path)
        
        # Run prediction
        Y_predicted = model.predict(np_img_input, verbose=0)
        
        class_ELA = ['Real', 'Tampered']
        accuracy = round(np.max(Y_predicted[0]) * 100)
        prediction = class_ELA[np.argmax(Y_predicted[0])]
        
        print(f"\nResult: {accuracy}% confidence of image being {prediction}")
        
        # Save ELA result for inspection
        ela_output_path = "user_ela_result.jpg"
        ela_img.save(ela_output_path)
        print(f"ELA image saved to: {ela_output_path}")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    # We'll use the original model since we are in the sandbox environment
    # where we already handled the TF/Keras versions earlier.
    model_file = 'ELA_Training/model_ela.h5'
    user_img = 'test_user_img.jpg'
    analyze_image(user_img, model_file)
