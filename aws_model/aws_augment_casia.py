import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
from PIL import ImageChops, ImageEnhance

def convert_to_ela_image(path, quality):
    try:
        filename = path
        resaved_filename = 'tempresaved.jpg'
        im = Image.open(filename)
        bm = im.convert('RGB')
        im.close()
        im = bm
        im.save(resaved_filename, 'JPEG', quality=quality)
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
        return ela_im
    except Exception as e:
        print(f"Error converting image {path}: {e}")
        return None

def build_image_list(path_to_image, label, images):
    for file in tqdm(os.listdir(path_to_image)):
        try:
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'tif')):
                if int(os.stat(os.path.join(path_to_image, file)).st_size) > 10000:
                    images.append(os.path.join(path_to_image, file) + ',' + label + '\n')
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return images

# Paths to the CASIA dataset
data_au = '/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Au'
data_tp = '/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Tp'

# Build dataset for CASIA
images = []
print(f"Total number of images collected initially: {len(images)}")

images = build_image_list(data_au, '0', images)  # Assuming '0' for authentic

# Shuffle and limit the number of authentic images to 5000
random.shuffle(images)
# images = images[:5000]
print(f"Total number of images collected authentic: {len(images)}")


images = build_image_list(data_tp, '1', images)  # Assuming '1' for tampered
print(f"Total number of images collected total: {len(images)}")

image_name = []
label = []
for i in tqdm(range(len(images))):
    try:
        img_path, lbl = images[i].split(',')
        image_name.append(img_path)
        label.append(lbl.strip())
    except Exception as e:
        print(f"Error parsing image list entry {images[i]}: {e}")

dataset = pd.DataFrame({'image': image_name, 'class_label': label})
dataset.to_csv('casia_dataset.csv', index=False)

dataset = pd.read_csv('casia_dataset.csv')
X = []
Y = []
for index, row in dataset.iterrows():
    try:
        ela_image = convert_to_ela_image(row['image'], 90)
        if ela_image:
            X.append(np.array(ela_image.resize((128, 128))).flatten() / 255.0)
            Y.append(int(row['class_label']))
    except Exception as e:
        print(f"Error processing image {row['image']}: {e}")

X = np.array(X)
Y = to_categorical(Y, 2)

X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

# Check the shape of training and validation data
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of validation samples: {X_val.shape[0]}")
# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Load the pre-trained model
model = load_model('/home/ubuntu/forgery/forgery/aws_model/aws_model_4_augmented.h5')

# Compile the model
optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
epochs = 20
batch_size = 100

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    validation_data=(X_val, Y_val),
    epochs=epochs, verbose=2, callbacks=[early_stopping]
)

print("Starting to save the model")
model.save("aws_model_4_augmented_casia_1.h5")
print("Ending after save the model")
