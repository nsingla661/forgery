import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Load the pre-trained model
model = load_model('model_casia_run1.h5')

def prepare_image(image_path):
    # Convert to ELA image and preprocess the same way as in training
    ela_image = convert_to_ela_image(image_path, 91)
    ela_image = ela_image.resize((128, 128))  # Resize to match training
    image_array = np.array(ela_image)
    image_array = image_array / 255.0  # Normalize the image
    return image_array


# Load and prepare the test image
image_path = 'Drivers-Forged (3) copy.jpg'
image = prepare_image(image_path)
image = image.reshape(1, 128, 128, 3)  # Add batch dimension

# Predict the values
Y_pred = model.predict(image)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Print the predicted class
print(f'Predicted class: {Y_pred_classes[0]}')



# Load and prepare the test image
image_path = 'photo_2024-09-10 13.46.53.jpeg'
image = prepare_image(image_path)
image = image.reshape(1, 128, 128, 3)  # Add batch dimension

# Predict the values
Y_pred = model.predict(image)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Print the predicted class
print(f'Predicted class for flash on document: {Y_pred_classes[0]}')

# Load and prepare the test image
image_path = 'image (9).png'
image = prepare_image(image_path)
image = image.reshape(1, 128, 128, 3)  # Add batch dimension

# Predict the values
Y_pred = model.predict(image)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Print the predicted class
print(f'Predicted class for image 9 : {Y_pred_classes[0]}')

# Load and prepare the test image
image_path = 'image (10).png'
image = prepare_image(image_path)
image = image.reshape(1, 128, 128, 3)  # Add batch dimension

# Predict the values
Y_pred = model.predict(image)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Print the predicted class
print(f'Predicted class: {Y_pred_classes[0]}')

# Load and prepare the test image
image_path = 's1.png'
image = prepare_image(image_path)
image = image.reshape(1, 128, 128, 3)  # Add batch dimension

# Predict the values
Y_pred = model.predict(image)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Print the predicted class
print(f'Predicted class of slack: {Y_pred_classes[0]}')

# Load and prepare the test image
image_path = 's2.png'
image = prepare_image(image_path)
image = image.reshape(1, 128, 128, 3)  # Add batch dimension

# Predict the values
Y_pred = model.predict(image)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Print the predicted class
print(f'Predicted class for blank image: {Y_pred_classes[0]}')


# Load and prepare the test image
image_path = 'p_camera.jpeg'
image = prepare_image(image_path)
image = image.reshape(1, 128, 128, 3)  # Add batch dimension

# Predict the values
Y_pred = model.predict(image)

# Convert predictions to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Print the predicted class
print(f'Predicted class for camera image: {Y_pred_classes[0]}')