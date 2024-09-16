import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Configuration
class Config:
    epochs = 30
    batch_size = 32
    lr = 1e-4
    image_size = (224, 224)  # ResNet50 expects (224, 224) images
    n_labels = 2


# Prepare Data
import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def convert_to_ela_image(path, quality):
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


def prepare_data():
    image_size = (224, 224)  # ResNet50 expects images of size 224x224

    X = []
    Y = []

    # Define paths to your dataset
    path_authentic = "data/input/casia-dataset/CASIA2/Au/"
    path_tampered = "data/input/casia-dataset/CASIA2/Tp/"

    def process_images(path, label):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.lower().endswith(("jpg", "png", "tif")):
                    full_path = os.path.join(dirname, filename)
                    try:
                        ela_image = convert_to_ela_image(full_path, 91)
                        if ela_image is not None:
                            ela_image = ela_image.resize(image_size)
                            X.append(np.array(ela_image))
                            Y.append(label)
                            if len(Y) % 500 == 0:
                                print(f"Processing {len(Y)} images")
                    except Exception as e:
                        print(f"Error processing image {full_path}: {e}")

    # Process authentic images (label 1)
    process_images(path_authentic, 1)
    # Process tampered images (label 0)
    process_images(path_tampered, 0)

    X = np.array(X)
    Y = to_categorical(Y, num_classes=2)

    return X, Y


# Use the function to prepare your data
X, Y = prepare_data()

# Convert labels to categorical format
Y = to_categorical(Y, Config.n_labels)

# Reshape X to fit the expected input size of ResNet50
X = X.reshape(-1, *Config.image_size, 3)

# Split data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

# Define data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode="nearest",
    validation_split=0.2,
)

datagen.fit(X_train)

train_generator = datagen.flow(
    X_train, Y_train, batch_size=Config.batch_size, subset="training"
)
validation_generator = datagen.flow(
    X_val, Y_val, batch_size=Config.batch_size, subset="validation"
)

# Load pre-trained ResNet50 model + higher level layers
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(Config.n_labels, activation="softmax")(x)

# Define the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
optimizer = Adam(learning_rate=Config.lr)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_accuracy", min_delta=0, patience=5, verbose=1, mode="max"
)

# Train the model
history = model.fit(
    train_generator,
    epochs=Config.epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping],
)

# Optionally, unfreeze some layers and fine-tune
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
    layer.trainable = True

# Recompile the model after unfreezing
model.compile(
    optimizer=Adam(learning_rate=Config.lr / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    epochs=Config.epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping],
)

# Save the model
print("Starting to save the model")
model.save("model_resnet50_finetuned.h5")
