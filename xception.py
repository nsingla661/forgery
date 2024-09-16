import os
import shutil
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import sys
from tempfile import NamedTemporaryFile

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import Xception

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = "casia-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F59500%2F115146%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240911%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240911T104634Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D67a5d0b3c194195cbd4639829f8330d73edb79354f6876f49ae86c3103d0a4a3e141cbe4f04d7d08419e9f4f7efd5d0b30e6583c7b85282878ab4fc429b5fd6f81e39a4b002ef66d1f7f514078d3fed8cb20ef39bcb4143492b3ca683a5f008923ff2ffe2e2ab887b0532196b6f313dca996fe4d487e8be2ec4ce1b4b2d528a2c2284d198192e08e2e77f7210a4536b25aa5766774eeca238d1f6866acff418c63b54301a61cea59df6f94614bbb65da2f41dea385576d3cae6dbf63267de0e5660d54f8a762ff33c474a0c8aac842a4b778ffb9ac27725050fb9ed3203f20c899afddba265a4528dbcf41d7f51c5d97da7e5bfeb21d0c7b29174a96c30c0d0e"
TMP_INPUT_PATH = "./data/input"
TMP_WORKING_PATH = "./data/working"

# Make directories in temporary paths where writing is allowed
os.makedirs(TMP_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(TMP_WORKING_PATH, 0o777, exist_ok=True)

# Process the data sources
for data_source_mapping in DATA_SOURCE_MAPPING.split(","):
    directory, download_url_encoded = data_source_mapping.split(":")
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(TMP_INPUT_PATH, directory)

    # Check if the dataset already exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)
        try:
            with urlopen(download_url) as fileres, NamedTemporaryFile(
                delete=False
            ) as tfile:
                total_length = fileres.headers.get("content-length")
                print(f"Downloading {directory}, {total_length} bytes compressed")

                dl = 0
                data = fileres.read(CHUNK_SIZE)
                while data:
                    dl += len(data)
                    tfile.write(data)
                    done = int(50 * dl / int(total_length))
                    sys.stdout.write(
                        f"\r[{'=' * done}{' ' * (50 - done)}] {dl} bytes downloaded"
                    )
                    sys.stdout.flush()
                    data = fileres.read(CHUNK_SIZE)

                # Uncompress file
                if filename.endswith(".zip"):
                    with ZipFile(tfile.name) as zfile:
                        zfile.extractall(destination_path)
                else:
                    with tarfile.open(tfile.name) as tarfile:
                        tarfile.extractall(destination_path)

                print(f"\nDownloaded and uncompressed: {directory}")

        except HTTPError as e:
            print(
                f"Failed to load (likely expired) {download_url} to path {destination_path}"
            )
            continue
        except OSError as e:
            print(f"Failed to load {download_url} to path {destination_path}")
            continue
    else:
        print(f"{directory} already exists. Skipping download.")

print("Data source import complete.")


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


# from keras.callbacks import EarlyS
from keras.callbacks import EarlyStopping

from PIL import Image, ImageChops, ImageEnhance


# def convert_to_ela_image(path, quality):
#     try:
#         temp_filename = "temp_file_name.jpg"
#         ela_filename = "temp_ela.png"

#         image = Image.open(path).convert("RGB")
#         image.save(temp_filename, "JPEG", quality=quality)
#         temp_image = Image.open(temp_filename)

#         ela_image = ImageChops.difference(image, temp_image)

#         extrema = ela_image.getextrema()
#         max_diff = max([ex[1] for ex in extrema])
#         if max_diff == 0:
#             max_diff = 1
#         scale = 255.0 / max_diff

#         ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

#         # Cleanup
#         os.remove(temp_filename)

#         return ela_image
#     except OSError as e:
#         print(f"Error processing file {path}: {e}")
#         return None
    

def convert_to_ela_image(path, quality):
    try:
        temp_filename = "temp_file_name.jpg"
        ela_filename = "temp_ela.png"

        # Open and convert the image to grayscale
        image = Image.open(path).convert("L")
        image.save(temp_filename, "JPEG", quality=quality)

        # Open the temporary image and convert it to grayscale
        temp_image = Image.open(temp_filename).convert("L")

        # Compute the ELA
        ela_image = ImageChops.difference(image, temp_image)

        # Get the extrema for scaling
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema if isinstance(ex, tuple)])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        # Enhance the brightness of the ELA image
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Save the ELA image (optional, for debugging or verification)
        ela_image.save(ela_filename)

        # Cleanup
        os.remove(temp_filename)

        return ela_image
    except OSError as e:
        print(f"Error processing file {path}: {e}")
        return None
    
# def convert_to_ela_image(path, quality):
#     try:
#         temp_filename = "temp_file_name.jpg"
#         ela_filename = "temp_ela.png"

#         image = Image.open(path).convert("L")  # Convert to grayscale
#         image.save(temp_filename, "JPEG", quality=quality)
#         temp_image = Image.open(temp_filename).convert("L")  # Ensure temp image is also grayscale

#         ela_image = ImageChops.difference(image, temp_image)

#         extrema = ela_image.getextrema()
        
#         # Ensure extrema has the expected format
#         if not extrema or not isinstance(extrema[0], tuple):
#             print(f"Unexpected extrema format: {extrema}")
#             max_diff = 1
#         else:
#             max_diff = max([ex[1] for ex in extrema if isinstance(ex, tuple) and len(ex) > 1])

#         if max_diff == 0:
#             max_diff = 1
#         scale = 255.0 / max_diff

#         ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

#         # Cleanup
#         os.remove(temp_filename)

#         return ela_image
#     except OSError as e:
#         print(f"Error processing file {path}: {e}")
#         return None


import tensorflow as tf


class Config:
    CASIA1 = os.path.abspath("./data/input/casia-dataset/CASIA1")
    CASIA2 = os.path.abspath("./data/input/casia-dataset/CASIA2")
    autotune = tf.data.experimental.AUTOTUNE
    epochs = 30
    batch_size = 32
    lr = 1e-3
    name = "xception"
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
    print("Contents:", list(path_to_check.glob("*")))
else:
    print("Directory does not exist.")


"""# Test on a Authentic image"""
import cv2
from os.path import join
from pathlib import Path
import random
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot

CASIA2 = Path("data/input/casia-dataset/CASIA2")

image_size = (128, 128)


def prepare_image(image_path, image_size=(128, 128)):
    ela_image = convert_to_ela_image(image_path, 91)
    if ela_image is None:
        return None
    return np.array(ela_image.resize(image_size)).flatten() / 255.0


X = []  # ELA converted images
Y = []  # 0 for fake, 1 for real


def is_image_corrupt(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify if image is corrupt
        return False
    except (IOError, SyntaxError) as e:
        print(f"Image {image_path} is corrupt: {e}")
        return True


import random
import numpy as np

path = CASIA2 / "Au/"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.lower().endswith(("jpg", "png", "tif")):
            full_path = os.path.join(dirname, filename)
            if not is_image_corrupt(full_path):
                result = prepare_image(full_path, image_size)
                if result is not None:
                    X.append(result)
                    Y.append(1)  # Assuming Y label as 0, adjust as needed
                    if len(Y) % 500 == 0:
                        print(f"Processing {len(Y)} images")

random.shuffle(X)
X = X[:3000]
Y = Y[:3000]
print("length of Authentic images used ")
print(len(X), len(Y))

path = CASIA2 / "Tp/"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.lower().endswith(("jpg", "png", "tif")):
            full_path = os.path.join(dirname, filename)
            if not is_image_corrupt(full_path):
                result = prepare_image(full_path, image_size)
                if result is not None:
                    X.append(result)
                    Y.append(0)  # Assuming Y label as 0, adjust as needed
                    if len(Y) % 500 == 0:
                        print(f"Processing {len(Y)} images")

print("length of authentic + tempered images")
print(len(X), len(Y))

import numpy as np

X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

from sklearn.model_selection import train_test_split

# Assuming X and Y are already defined

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

print("training data set : ")
print(len(X_train), len(Y_train))
print("print validation data set : ")
print(len(X_val), len(Y_val))


def build_model():
    base_model = Xception(
        include_top=False, weights='imagenet', input_shape=(128, 128, 3)
    )
    # Add custom layers on top of Xception
    model = Sequential()
    model.add(base_model)  # Add the Xception model
    model.add(Flatten())  # Flatten the output of Xception
    model.add(Dense(256, activation="relu"))  # Dense layer
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(2, activation="softmax"))  # Final layer for 2 classes (authentic/forged)

    return model



model = build_model()
model.summary()

from keras import optimizers

model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=["accuracy"])

epochs = 2
batch_size = 32

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your initial learning rate
init_lr = 1e-4

optimizer = Adam(lr = init_lr, decay = init_lr/epochs) 
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(
    monitor="val_accuracy", min_delta=0, patience=5, verbose=0, mode="auto"
)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

x_train2 = np.array(X_train, copy=True)
y_train2 = np.array(Y_train, copy=True)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode="nearest",
    validation_split=0.2,
)

datagen.fit(X_train)

validation_generator = datagen.flow(
    x_train2, y_train2, batch_size=32, subset="validation"
)
train_generator = datagen.flow(x_train2, y_train2, batch_size=32, subset="training")

# Ensure correct metric name and mode for early stopping
early_stopping = EarlyStopping(
    monitor="val_accuracy", min_delta=0, patience=5, verbose=0, mode="auto"
)
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping],
)

# model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# hist = model.fit(
#     X_train,
#     Y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(X_val, Y_val),
#     callbacks=[early_stopping],
# )


print("starting to save the model")
model.save("model_xception.h5")
