import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os

# Load the pre-trained model (replace with the correct path to your model file)
model = load_model('model_xception_3.h5')

# Function to convert an image to ELA (Error Level Analysis)
def convert_to_ela_image(path, quality):
    try:
        temp_filename = "temp_file_name.jpg"
        ela_filename = "temp_ela.png"

        image = Image.open(path).convert("RGB")
        image.save(temp_filename, "JPEG", quality=quality)
        temp_image = Image.open(temp_filename)

        ela_image = ImageChops.difference(image, temp_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Cleanup
        os.remove(temp_filename)

        return ela_image
    except OSError as e:
        print(f"Error processing file {path}: {e}")
        return None

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
    print(prediction)
    
    # Convert prediction to label (assuming binary classification)
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    if predicted_label == 0:
        return "Authentic"
    else:
        return "Forged"

# Example usage:
image_path = 'p_camera.jpeg'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")




# Example usage:
image_path = 'Drivers-Forged (3) copy.jpg'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")


# Example usage:
image_path = 'image (9).png'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")


# Example usage:
image_path = 'image (9).png'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")


# Example usage:
image_path = 'photo_2024-09-10 13.46.53.jpeg'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")


# Example usage:
image_path = 'Robert_Forged.jpg'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")


# Example usage:
image_path = 's1.png'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")


# Example usage:
image_path = 'Untitled design-2.png'  # Replace with your image path
result = predict_image(model, image_path, image_size=(128, 128))
print(f"The image {image_path} is predicted to be: {result}")

print("now au folder images")

au_folder_path = "/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Au"

# Get the first 5 images from the "Au" folder
image_files = os.listdir(au_folder_path)
image_files = [f for f in image_files if f.endswith(('.png', '.jpg', '.jpeg'))][:5]  # Pick 5 images


# Predict on each image
for image_file in image_files:
    image_path = os.path.join(au_folder_path, image_file)
    
    # Preprocess the image
    processed_image = preprocess_image_for_prediction(image_path)
    
    # Make the prediction
    prediction = model.predict(processed_image)
    
    # Interpret the prediction (Assuming binary classification)
    predicted_label = np.argmax(prediction, axis=1)[0]
    if predicted_label == 0:
        result = "Authentic"
    else:
        result = "Forged"
    
    # Print the result
    print(f"Prediction for {image_file}: {result}")