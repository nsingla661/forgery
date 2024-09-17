import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
from PIL import Image, ImageChops, ImageEnhance
import os

# Load the pre-trained model
model = load_model("model_casia_regularised.h5")

def convert_to_ela_image(path, quality):
    temp_filename = "temp_file_name.jpg"

    try:
        # Open the original image
        image = Image.open(path).convert("RGB")

        # Save the image with lower quality to create a temporary JPEG
        image.save(temp_filename, "JPEG", quality=quality)

        # Open the newly saved image
        temp_image = Image.open(temp_filename)

        # Compute the ELA image by finding the difference
        ela_image = ImageChops.difference(image, temp_image)

        # Scale the ELA image
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return ela_image

    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return None

    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)



def prepare_image(image_path):
    # Convert to ELA image and preprocess the same way as in training
    ela_image = convert_to_ela_image(image_path, 91)
    ela_image = ela_image.resize((128, 128))  # Resize to match training
    image_array = np.array(ela_image)
    image_array = image_array / 255.0  # Normalize the image
    return image_array


# Function to predict and print the results
def predict_and_print(image_path):
    # Load and prepare the test image
    image = prepare_image(image_path)
    image = image.reshape(1, 128, 128, 3)  # Add batch dimension

    # Predict the values
    Y_pred = model.predict(image)

    # Print the prediction probabilities
    print(f"Prediction probabilities for {image_path}: {Y_pred}")

    # Convert predictions to class labels
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    # Print the predicted class
    print(f"Predicted class for {image_path}: {Y_pred_classes[0]}")


# Test the function on different images
predict_and_print("Drivers-Forged (3) copy.jpg")
predict_and_print("photo_2024-09-10 13.46.53.jpeg")
predict_and_print("image (9).png")
predict_and_print("image (10).png")
predict_and_print("s1.png")
predict_and_print("s2.png")
predict_and_print("p_camera.jpeg")


def evaluate_model_on_known_authentic_images(model, image_paths):
    global count
    count = 0
    for image_path in image_paths:
        # Load and prepare the test image
        image = prepare_image(image_path)
        image = image.reshape(1, 128, 128, 3)  # Add batch dimension

        # Predict the values
        Y_pred = model.predict(image)

        # Print the prediction probabilities
        # print(f'Prediction probabilities for {image_path}: {Y_pred}')

        # Convert predictions to class labels
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        if Y_pred_classes == 1:
            count = count + 1
        # Print the predicted class
        # print(f'Predicted class for {image_path}: {Y_pred_classes[0]}')


# Get the list of first 10 images in the 'Au' directory
au_directory = "data/input/casia-dataset/CASIA2/Au"
authentic_image_paths = [
    os.path.join(au_directory, fname) for fname in os.listdir(au_directory)[:1000]
]

evaluate_model_on_known_authentic_images(model, authentic_image_paths)
print(f"total count of correct indicates are : {count}")
