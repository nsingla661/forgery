import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
import os

# Function to preprocess and prepare the image for prediction
def preprocess_image_for_prediction(image_path, image_size=(128, 128), quality=91):
    """
    Preprocess the input image by applying ELA and resizing it to the target size.
    
    Args:
        image_path (str): Path to the image to predict.
        image_size (tuple): Desired image size (width, height).
        quality (int): Quality of the image for ELA.
        
    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    ela_image = convert_to_ela_image(image_path, quality)
    
    if ela_image is None:
        return None
    
    # Resize the ELA image and scale pixel values to the range [0, 1]
    processed_image = ela_image.resize(image_size)
    image_array = np.array(processed_image).astype('float32') / 255.0

    # Expand dimensions to match the input shape required by the model
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    return image_array

# Function to make predictions on an image
def predict_image(model, image_path, image_size=(128, 128)):
    """
    Predict whether the image is authentic or forged using the trained model.
    
    Args:
        model: Trained Keras model.
        image_path (str): Path to the image to predict.
        image_size (tuple): Desired image size (width, height) for prediction.
        
    Returns:
        str: Prediction result ('Authentic' or 'Forged').
    """
    # Preprocess the image
    processed_image = preprocess_image_for_prediction(image_path, image_size)
    
    if processed_image is None:
        return "Error: Could not preprocess image."

    # Predict the class using the trained model
    prediction = model.predict(processed_image)
    
    # Convert prediction to label (assuming binary classification)
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    if predicted_label == 0:
        return "Authentic"
    else:
        return "Forged"

# Example usage:
image_path = 'forgery/p_camera.jpeg'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image is predicted to be: {result}")
