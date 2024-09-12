import os
import shutil
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import sys
from tempfile import NamedTemporaryFile

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'casia-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F59500%2F115146%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240911%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240911T104634Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D67a5d0b3c194195cbd4639829f8330d73edb79354f6876f49ae86c3103d0a4a3e141cbe4f04d7d08419e9f4f7efd5d0b30e6583c7b85282878ab4fc429b5fd6f81e39a4b002ef66d1f7f514078d3fed8cb20ef39bcb4143492b3ca683a5f008923ff2ffe2e2ab887b0532196b6f313dca996fe4d487e8be2ec4ce1b4b2d528a2c2284d198192e08e2e77f7210a4536b25aa5766774eeca238d1f6866acff418c63b54301a61cea59df6f94614bbb65da2f41dea385576d3cae6dbf63267de0e5660d54f8a762ff33c474a0c8aac842a4b778ffb9ac27725050fb9ed3203f20c899afddba265a4528dbcf41d7f51c5d97da7e5bfeb21d0c7b29174a96c30c0d0e'
TMP_INPUT_PATH = './data/input'
TMP_WORKING_PATH = './data/working'

# Make directories in temporary paths where writing is allowed
os.makedirs(TMP_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(TMP_WORKING_PATH, 0o777, exist_ok=True)

# Process the data sources
for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(TMP_INPUT_PATH, directory)
    
    # Check if the dataset already exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)
        try:
            with urlopen(download_url) as fileres, NamedTemporaryFile(delete=False) as tfile:
                total_length = fileres.headers.get('content-length')
                print(f'Downloading {directory}, {total_length} bytes compressed')
                
                dl = 0
                data = fileres.read(CHUNK_SIZE)
                while data:
                    dl += len(data)
                    tfile.write(data)
                    done = int(50 * dl / int(total_length))
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {dl} bytes downloaded")
                    sys.stdout.flush()
                    data = fileres.read(CHUNK_SIZE)
                    
                # Uncompress file
                if filename.endswith('.zip'):
                    with ZipFile(tfile.name) as zfile:
                        zfile.extractall(destination_path)
                else:
                    with tarfile.open(tfile.name) as tarfile:
                        tarfile.extractall(destination_path)
                        
                print(f'\nDownloaded and uncompressed: {directory}')
        
        except HTTPError as e:
            print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
            continue
        except OSError as e:
            print(f'Failed to load {download_url} to path {destination_path}')
            continue
    else:
        print(f'{directory} already exists. Skipping download.')

print('Data source import complete.')


"""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
"""

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#from keras.callbacks import EarlyS
from keras.callbacks import EarlyStopping

from PIL import Image, ImageChops, ImageEnhance
import os
import itertools

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

import tensorflow as tf
class Config:
    CASIA1 = os.path.abspath("./data/input/casia-dataset/CASIA1")
    CASIA2 = os.path.abspath("./data/input/casia-dataset/CASIA2")
    autotune = tf.data.experimental.AUTOTUNE
    epochs = 30
    batch_size = 32
    lr = 1e-3
    name = 'xception'
    n_labels = 2
    image_size = (224, 224)
    decay = 1e-6
    momentum = 0.95
    nesterov = False

from pathlib import Path

# Print the absolute path for verification
path_to_check = Path(os.path.abspath("./data/input/casia-dataset/CASIA2"))
print(f"Checking directory: {path_to_check}")
if path_to_check.exists() and path_to_check.is_dir():
    print("Directory exists.")
    print("Contents:", list(path_to_check.glob('*')))
else:
    print("Directory does not exist.")



def compute_ela_cv(path, quality):
    temp_filename = 'temp_file_name.jpg'
    SCALE = 15
    orig_img = cv2.imread(path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(temp_filename, orig_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    compressed_img = cv2.imread(temp_filename)

    diff = SCALE * cv2.absdiff(orig_img, compressed_img)
    return diff

def random_sample(path, extension=None):
    if extension:
        items = list(Path(path).glob(f'*.{extension}'))
    else:
        items = list(Path(path).glob('*'))

    if not items:
        raise ValueError(f"No files found in {path} with extension {extension}")

    p = random.choice(items)
    return p.as_posix()

"""# Test on a Authentic image"""

import cv2
from os.path import join
from pathlib import Path
import random
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot

# Assuming Config.CASIA2 and random_sample() are defined elsewhere in your code

# Your existing code continues from here
p = join(Config.CASIA2, 'Au/')
p = random_sample(p)
orig = cv2.imread(p)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0
init_val = 100
columns = 3
rows = 3

fig = plt.figure(figsize=(15, 10))
for i in range(1, columns * rows + 1):
    quality = init_val - (i - 1) * 3
    # Assuming compute_ela_cv is defined elsewhere in your code
    img = compute_ela_cv(path=p, quality=quality)
    if i == 1:
        img = orig.copy()
    ax = fig.add_subplot(rows, columns, i)
    ax.title.set_text(f'q: {quality}')
    plt.imshow(img)
plt.show()

"""# Test on a tampered fake image"""

try:
    p = join(Config.CASIA2, 'Tp/')
    p = random_sample(p)
    orig = cv2.imread(p)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0
except ValueError as e:
    print(e)
init_val = 100
columns = 3
rows = 3

fig=plt.figure(figsize=(15, 10))
for i in range(1, columns*rows +1):
    quality=init_val - (i-1) * 3
    img = compute_ela_cv(path=p, quality=quality)
    if i == 1:
        img = orig.copy()
    ax = fig.add_subplot(rows, columns, i)
    ax.title.set_text(f'q: {quality}')
    plt.imshow(img)
plt.show()

CASIA2 = Path("data/input/casia-dataset/CASIA2")

real_image_path = CASIA2/"Au/Au_ani_00001.jpg"
Image.open(real_image_path)

convert_to_ela_image(real_image_path, 91)

fake_image_path = CASIA2/'Tp/Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg'
Image.open(fake_image_path)

convert_to_ela_image(fake_image_path, 91)

"""DENOISING REAL IMAGE"""

# Color-image denoising
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.util import random_noise
import skimage.io

img_r1 = skimage.io.imread(CASIA2/'Au/Au_ani_00001.jpg')
img_r = skimage.img_as_float(img_r1)  # Converting image as float

sigma_est = estimate_sigma(img_r, channel_axis=-1, average_sigmas=True)  # Noise estimation

# Denoising using Bayes
img_bayes = denoise_wavelet(img_r, method='BayesShrink', mode='soft', wavelet_levels=3,
                             wavelet='coif5', channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)

# Denoising using Visushrink
img_visushrink = denoise_wavelet(img_r, method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5,
                                  wavelet='coif5', channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)

import cv2
psnr_noisy = cv2.PSNR(img_r,img_r)
psnr_noisy

psnr_bayes = cv2.PSNR(img_r,img_bayes)
psnr_bayes

psnr_visu = cv2.PSNR(img_r,img_visushrink)
psnr_visu

# Plotting images
plt.figure(figsize=(30,30))

plt.subplot(2,2,1)
plt.imshow(img_r1,cmap=plt.cm.gray)
plt.title('Original Image',fontsize=30)

plt.subplot(2,2,2)
plt.imshow(img_r,cmap=plt.cm.gray)
plt.title('Noisy Image',fontsize=30)

plt.subplot(2,2,3)
plt.imshow(img_bayes,cmap=plt.cm.gray)
plt.title('Denoising using Bayes',fontsize=30)

plt.subplot(2,2,4)
plt.imshow(img_visushrink,cmap=plt.cm.gray)
plt.title('Denoising using Visushrink',fontsize=30)

plt.show()

print('PSNR[Original vs. Noisy Image]', psnr_noisy)
print('PSNR[Original vs. Denoised(VisuShrink)]', psnr_visu)
print('PSNR[Original vs. Denoised(Bayes)]', psnr_bayes)

"""DENOISING FAKE IMAGE"""

from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.util import random_noise
import skimage.io

img_f = skimage.io.imread(CASIA2/'Tp/Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg')
img_f = skimage.img_as_float(img_f)  # Converting image as float

sigma = 0.35  # Noise
imgn = random_noise(img_f, var=sigma**2)  # Adding noise

sigma_est = estimate_sigma(img_f, channel_axis=-1, average_sigmas=True)  # Noise estimation

# Denoising using Bayes
img_bayes = denoise_wavelet(img_f, method='BayesShrink', mode='soft', wavelet_levels=3,
                             wavelet='coif5', channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)

# Denoising using Visushrink
img_visushrink = denoise_wavelet(img_f, method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5,
                                  wavelet='coif5', channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)

import cv2
psnr_noisy = cv2.PSNR(img_f,img_f)
psnr_noisy

psnr_bayes = cv2.PSNR(img_f,img_bayes)
psnr_bayes

psnr_visu = cv2.PSNR(img_f,img_visushrink)
psnr_visu

# Plotting images
plt.figure(figsize=(30,30))

plt.subplot(2,2,1)
plt.imshow(img_f,cmap=plt.cm.gray)
plt.title('Original Image',fontsize=30)

plt.subplot(2,2,2)
plt.imshow(img_f,cmap=plt.cm.gray)
plt.title('Noisy Image',fontsize=30)

plt.subplot(2,2,3)
plt.imshow(img_bayes,cmap=plt.cm.gray)
plt.title('Original Image',fontsize=30)

plt.subplot(2,2,4)
plt.imshow(img_visushrink,cmap=plt.cm.gray)
plt.title('Original Image',fontsize=30)

plt.show()

print('PSNR[Original vs. Noisy Image]', psnr_noisy)
print('PSNR[Original vs. Denoised(VisuShrink)]', psnr_visu)
print('PSNR[Original vs. Denoised(Bayes)]', psnr_bayes)

# Color-image denoising
from skimage.restoration import (denoise_wavelet,estimate_sigma)
from skimage.util import random_noise
# from sklearn.metrics import peak_signal_noise_ratio
import skimage.io

def denoise_img(img):
    #img=skimage.io.imread('../input/casia-dataset/CASIA2/Tp/Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg')
    img=skimage.img_as_float(img_f) #converting image as float


    sigma_est=estimate_sigma(img,multichannel=True,average_sigmas=True)  #Noise estimation

    # Denoising using Bayes
    img_bayes=denoise_wavelet(img,method='BayesShrink',mode='soft',wavelet_levels=3,
                          wavelet='coif5',multichannel=True,convert2ycbcr=True,rescale_sigma=True)


    #Denoising using Visushrink
    img_visushrink=denoise_wavelet(img,method='VisuShrink',mode='soft',sigma=sigma_est/3,wavelet_levels=5,
    wavelet='coif5',multichannel=True,convert2ycbcr=True,rescale_sigma=True)
    return img_bayes

image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 91).resize(image_size)).flatten() / 255.0

X = [] # ELA converted images
Y = [] # 0 for fake, 1 for real

import random
import numpy as np
path = CASIA2/'Au/'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

random.shuffle(X)
X = X[:2100]
Y = Y[:2100]
print(len(X), len(Y))

path = CASIA2/'Tp/'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

print(len(X), len(Y))

import numpy as np
X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

from sklearn.model_selection import train_test_split

# Assuming X and Y are already defined

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)
X = X.reshape(-1, 1, 1, 1)

print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))

def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    return model

model = build_model()
model.summary()

from keras import optimizers
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

epochs = 24
batch_size = 32

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your initial learning rate
init_lr = 1e-4

# Define your optimizer without using deprecated arguments
optimizer = Adam(learning_rate=init_lr)

early_stopping = EarlyStopping(monitor = 'val_acc',
                              min_delta = 0,
                              patience = 2,
                              verbose = 0,
                              mode = 'auto')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

x_train2 = np.array(X_train, copy=True)
y_train2 = np.array(Y_train, copy=True)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode='nearest',
    validation_split=0.2
)

datagen.fit(X_train)

validation_generator = datagen.flow(x_train2, y_train2, batch_size=32, subset='validation')
train_generator = datagen.flow(x_train2, y_train2, batch_size=32, subset='training')

# Ensure correct metric name and mode for early stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Correct metric name
    min_delta=0,
    patience=30,
    verbose=1,
    mode='max'  # Use 'max' for accuracy
)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping]
)

model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

hist = model.fit(X_train,
                 Y_train,
                 batch_size = batch_size,
                 epochs = epochs,
                validation_data = (X_val, Y_val),
                callbacks = [early_stopping])

model.save('model_casia_run1.h5')

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.metrics import confusion_matrix

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred,axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))

class_names = ['fake', 'real']

real_image_path = CASIA2/'Au/Au_ani_00040.jpg'
Image.open(real_image_path)

image = prepare_image(real_image_path)
image = image.reshape(-1, 128, 128, 3)
y_pred = model.predict(image)
y_pred_class = np.argmax(y_pred, axis = 1)[0]
print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

fake_image_path = CASIA2 / 'Tp' / 'Tp_D_NRD_S_N_ani00041_ani00040_00161.tif'
Image.open(fake_image_path)

image = prepare_image(fake_image_path)
image = image.reshape(-1, 128, 128, 3)
y_pred = model.predict(image)
y_pred_class = np.argmax(y_pred, axis = 1)[0]
print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

fake_image = os.listdir(CASIA2 / 'Tp')

correct = 0
total = 0
for file_name in fake_image:
    if file_name.endswith('jpg') or filename.endswith('png'):
        fake_image_path = CASIA2 / 'Tp' / file_name
        image = prepare_image(fake_image_path)
        image = image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(image)
        y_pred_class = np.argmax(y_pred, axis = 1)[0]
        total += 1
        if y_pred_class == 0:
            correct += 1

print(f'Total: {total}, Correct: {correct}, Acc: {correct / total * 100.0}')

real_image = os.listdir('../input/casia-dataset/CASIA2/Au/')
correct_r = 0
total_r = 0
for file_name in real_image:
    if file_name.endswith('jpg') or filename.endswith('png'):
        real_image_path = CASIA2 / 'Au' / file_name
        image = prepare_image(real_image_path)
        image = image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(image)
        y_pred_class = np.argmax(y_pred, axis = 1)[0]
        total_r += 1
        if y_pred_class == 1:
            correct_r += 1

correct += correct_r
total += total_r
print(f'Total: {total_r}, Correct: {correct_r}, Acc: {correct_r / total_r * 100.0}')
print(f'Total: {total}, Correct: {correct}, Acc: {correct / total * 100.0}')

image_path1 = '/kaggle/input/casia-dataset/CASIA1/Au/Au_ani_0028.jpg'
Image.open(image_path1)

image = prepare_image(image_path1)
image = image.reshape(-1, 128, 128, 3)
y_pred = model.predict(image)
y_pred_class = np.argmax(y_pred, axis = 1)[0]
print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

image_path2 = '/kaggle/input/casia-dataset/CASIA1/Sp/Sp_D_NNN_A_ani0028_pla0007_0284.jpg'
Image.open(image_path2)

image = prepare_image(image_path2)
image = image.reshape(-1, 128, 128, 3)
y_pred = model.predict(image)
y_pred_class = np.argmax(y_pred, axis = 1)[0]
print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

convert_to_ela_image(image_path1, 91)

image_1_ELA=convert_to_ela_image(image_path1, 91)

convert_to_ela_image(image_path2, 91)

image_2_ELA=convert_to_ela_image(image_path2, 91)

ela_image = ImageChops.difference(image_1_ELA, image_2_ELA)

ela_image

print(y_pred_class)

def find_manipulated_region(ela, threshold=50):
    mask = np.array(ela) > threshold

    # Find the bounding box of the masked region
    if np.any(mask):
        coords = np.argwhere(mask)
        return coords
    else:
        return None

def make_pixels_white(img, white_coords):
    width, height = img.size
    black_img = Image.new('RGB', (width, height), color='black')
    img_arr = np.array(img)
    black_arr = np.array(black_img)
    for coord in white_coords:
        x, y, z = coord
        black_arr[x,y,:] = [255,255,255]
    mask = np.all(black_arr == [255,255,255], axis=-1)
    img_arr[mask] = [255,255,255]
    new_img = Image.fromarray(img_arr)
    return new_img

if y_pred_class==0:
    ela=convert_to_ela_image(image_path2,91)
    coords=find_manipulated_region(ela)
    modify_boundary=make_pixels_white(ela,coords)
    modify_boundary.show()
