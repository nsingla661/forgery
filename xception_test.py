import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os

# Load the pre-trained model (replace with the correct path to your model file)
model = load_model('model_xception_3.h5')
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os

def prepare_image_for_prediction(image_path, image_size=(128, 128)):
    """Preprocess the input image for prediction."""
    ela_image = convert_to_ela_image(image_path, 91)  # ELA preprocessing
    if ela_image is None:
        return None
    
    # Resize and normalize the image for the model
    ela_resized = ela_image.resize(image_size)
    ela_array = np.array(ela_resized) / 255.0  # Normalize to [0, 1] range
    
    # Reshape for the model input
    ela_array = ela_array.reshape(1, 128, 128, 3)  # Shape (1, 128, 128, 3)
    return ela_array

def convert_to_ela_image(path, quality):
    """Convert the input image to its ELA (Error Level Analysis) representation."""
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

        # Cleanup temporary file
        os.remove(temp_filename)

        return ela_image
    except OSError as e:
        print(f"Error processing file {path}: {e}")
        return None


# Example usage:
# image_path = 'p_camera.jpeg'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")




# # Example usage:
# image_path = 'Drivers-Forged (3) copy.jpg'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")


# # Example usage:
# image_path = 'image (9).png'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")


# # Example usage:
# image_path = 'image (9).png'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")


# # Example usage:
# image_path = 'photo_2024-09-10 13.46.53.jpeg'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")


# # Example usage:
# image_path = 'Robert_Forged.jpg'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")


# # Example usage:
# image_path = 's1.png'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")


# # Example usage:
# image_path = 'Untitled design-2.png'  # Replace with your image path
# result = predict_image(model, image_path, image_size=(128, 128))
# print(f"The image {image_path} is predicted to be: {result}")

print("now au folder images")

au_image_folder = "/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Au"

# Get the first 5 images from the "Au" folder
# image_files = os.listdir(au_folder_path)
# image_files = [f for f in image_files if f.endswith(('.png', '.jpg', '.jpeg'))][:500]  # Pick 5 images

count = 0


# Get a list of 5 sample images from the folder
image_paths = [os.path.join(au_image_folder, image_name) 
               for image_name in os.listdir(au_image_folder)[:5]]

# Iterate over the sample images, preprocess, and predict
for image_path in image_paths:
    # Preprocess the image for prediction
    processed_image = prepare_image_for_prediction(image_path)
    
    if processed_image is None:
        print(f"Skipping {image_path} due to preprocessing error.")
        continue

    # Make a prediction using the loaded model
    prediction = model.predict(processed_image)
    print(prediction)
    # Map prediction to class labels
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_labels = {0: "Authentic", 1: "Forged"}
    predicted_label = class_labels[predicted_class]

    print(f"Image: {image_path} | Predicted: {predicted_label}")
