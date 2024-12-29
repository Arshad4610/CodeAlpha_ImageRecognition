# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 13:06:11 2024

@author: rawqa
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')

def classify_image(img_path):
    """
    Classify an image using the MobileNetV2 model pre-trained on ImageNet.
    Args:
        img_path (str): Path to the image to be classified.
    """
    # Check if the file exists
    if not os.path.exists(img_path):
        print("File not found. Check the path.")
        return
    
    # Load and preprocess the image
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess image for MobileNetV2
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Make predictions
    predictions = model.predict(img_array)

    # Decode and print the top 3 predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score * 100:.2f}%)")

# Example usage
image_path = r"C:\Users\rawqa\OneDrive\Pictures\dog.jpg"  # Replace with your actual path
if os.path.exists(image_path):
    print("File found! Ready for classification.")
    classify_image(image_path)
else:
    print("File not found. Check the path.")
