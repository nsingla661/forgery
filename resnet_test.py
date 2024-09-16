import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os

# Configuration
class Config:
    image_size = (224, 224)  # ResNet50 expects images of size 224x224
    model_path = "model_resnet50_finetuned_complete.h5"  # Path to the saved model

def convert_to_ela_image(path, quality):
    """Convert an image to ELA format."""
    try:
        temp_filename = "temp_file_name.jpg"

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

def preprocess_image(image_path):
    """Load and preprocess the image for prediction."""
    image_size = Config.image_size  # ResNet50 expects images of size 224x224
    
    # Convert to ELA image
    ela_image = convert_to_ela_image(image_path, 91)
    if ela_image is not None:
        # Resize image to fit model input
        ela_image = ela_image.resize(image_size)
        
        # Convert image to numpy array and normalize
        ela_image_array = np.array(ela_image).astype('float32') / 255.0
        
        # Add batch dimension
        ela_image_array = np.expand_dims(ela_image_array, axis=0)
        return ela_image_array
    else:
        raise ValueError("Error processing image for ELA conversion.")

def predict_image(image_path):
    """Predict the class of an image given its file path."""
    # Load the trained model
    model = load_model(Config.model_path)
    
    # Preprocess the image
    image_array = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(image_array)
    
    # Convert prediction to label
    predicted_label = np.argmax(predictions, axis=1)
    
    # Return the predicted label
    return predicted_label[0]

# Example usage
if __name__ == "__main__":
    try:
        image_path = "image (9).png"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "image (10).png"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "Drivers-Forged (3) copy.jpg"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "Robert_Forged.jpg"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "Untitled design-2.png"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "p_camera.jpeg"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "s1.png"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "s2.png"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
        image_path = "photo_2024-09-10 13.46.53.jpeg"
        label = predict_image(image_path)
        print(f"The predicted label for the {image_path} image is: {label}")
    
    
    except Exception as e:
        print(f"An error occurred: {e}")
