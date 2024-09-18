import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

# Load the trained model
model = load_model('/home/ubuntu/forgery/forgery/aws_model/aws_model_4_casia_augmented.h5')

def convert_to_ela_image(path, quality=90):
    resaved_filename = 'tempresaved.jpg'
    im = Image.open(path)
    
    # Convert RGBA to RGB if necessary
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return ela_im

def preprocess_image(image_path):
    # Convert to ELA and resize
    ela_image = convert_to_ela_image(image_path).resize((128, 128))
    ela_image = np.array(ela_image) / 255.0  # Normalize pixel values
    ela_image = ela_image.reshape(1, 128, 128, 3)  # Reshape for model
    return ela_image

def predict_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Predict using the model
    prediction = model.predict(processed_image)
    # print(prediction)
    class_label = np.argmax(prediction, axis=1)
    
    return class_label[0]

# Paths to the folders
camera_clicked_path = '/home/ubuntu/forgery/forgery/camera_clicked/'
license_docs_path = '/home/ubuntu/forgery/forgery/license_docs/'
data_au = '/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Au'
data_tp = '/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Tp'

def predict_folder(folder_path):
    count = 0 
    total = 0
    for file in os.listdir(folder_path) :
        if file.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(folder_path, file)
            class_label = predict_image(image_path)
            print(class_label)
            count += (class_label)
            total += 1
            # print(f"Image: {file}, Predicted Class: {class_label}")
    print(f"Predicting for images in folder: {folder_path} , count : {count} , total = {total}")

# Run predictions for both folders

predict_folder(data_tp)

# camera_clicked_path = '/home/ubuntu/forgery/forgery/camera_clicked/'
# license_docs_path = '/home/ubuntu/forgery/forgery/license_docs/'
# predict_folder(camera_clicked_path)
# print(" ")
# print(" ")
# print(" ")
# predict_folder(license_docs_path)
