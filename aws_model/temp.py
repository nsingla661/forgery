import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
# import random

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop


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
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'tif')):
                if int(os.stat(os.path.join(path_to_image, file)).st_size) > 10000:
                    images.append(os.path.join(path_to_image, file) + ',' + label + '\n')
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return images

custom_path_original = 'images/training/original/'
custom_path_tampered = 'images/training/forged/'
data_au = '/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Au'
data_tp = '/home/ubuntu/forgery/forgery/data/input/casia-dataset/CASIA2/Tp'

training_data_set = 'dataset.csv'

images = []
images = build_image_list(custom_path_original, '0', images)
images = build_image_list(custom_path_tampered, '1', images)
count_len = len(images)
import random

temp1 = []
temp2 = []
temp1 = build_image_list(data_au, '0', temp1)  # Assuming '0' for authentic 
random.shuffle(temp1)
temp2 = build_image_list(data_tp, '1', temp2)  # Assuming '1' for tampered
random.shuffle(temp2)


print(f"the length of temp1 dataset is : {len(temp1)}")
print(f"the length of temp2 dataset is : {len(temp2)}")

images.extend(temp1[:200])
images.extend(temp2[:200])

print(f"the length of images dataset is : {count_len}")
print(f"the length of images dataset is : {len(images)}")
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
    X.append(array(convert_to_ela_image(row[0], 80).resize((128, 128))).flatten() / 255.0)
    Y.append(row[1])
X = np.array(X)
Y = to_categorical(Y, 2)


X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'valid', 
                 activation ='relu', input_shape = (128,128,3)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'valid', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))

model.summary()

optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 20
batch_size = 100

early_stopping = EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val), verbose = 2, callbacks=[early_stopping])


print("starting to save the model")
model.save("aws_model_7.h5")
print("ending after save the model")
