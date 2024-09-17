import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from PIL import Image
from pylab import *
from PIL import Image, ImageChops, ImageEnhance

from datetime import datetime
import cv2

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = 'tempresaved.jpg'
    im = Image.open(filename)
    bm = im.convert('RGB')
    im.close()
    im=bm
    im.save(resaved_filename, 'JPEG', quality = quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    im.close()
    bm.close()
    resaved_im.close()
    del filename
    del resaved_filename
    del im
    del bm
    del resaved_im
    del extrema
    del max_diff
    del scale
    return ela_im

def build_image_list(path_to_image, label, images):
    for file in tqdm(os.listdir(path_to_image)):
        try:
            if file.endswith('jpg') or file.endswith('JPG') or file.endswith('jpeg') or file.endswith('JPEG'):
                if int(os.stat(path_to_image + file).st_size) > 10000:
                    line = path_to_image + file  + ',' + label + '\n'
                    images.append(line)
        except:
            print(path_to_image + file)
    return images
    

custom_path_original = 'images/training/original/'
custom_path_tampered = 'images/training/forged/'

training_data_set = 'dataset.csv'

images = []
images = build_image_list(custom_path_original, '0', images)
images = build_image_list(custom_path_tampered, '1', images)

image_name = []
label = []
for i in tqdm(range(len(images))):
    image_name.append(images[i][0:-3])
    label.append(images[i][-2])

dataset = pd.DataFrame({'image':image_name,'class_label':label})
dataset.to_csv(training_data_set,index=False)

dataset = pd.read_csv('dataset.csv')
X = []
Y = []
for index, row in dataset.iterrows():
    X.append(array(convert_to_ela_image(row[0], 90).resize((128, 128))).flatten() / 255.0)
    Y.append(row[1])
X = np.array(X)
Y = to_categorical(Y, 2)


X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)

# Load the pre-trained model
model = load_model('aws_model_4_augmented_casia.h5')

# Compile the model
optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Retrain the model
epochs = 20
batch_size = 100

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping]
)

print("Starting to save the retrained model")
model.save("aws_model_4_augmented_retrained.h5")
print("Model saved successfully")
