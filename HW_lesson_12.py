import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train")
TEST_PATH  = os.path.join(BASE_DIR, "data", "test")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH, image_size=(128, 128), batch_size=32,
    label_mode="categorical"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH, image_size=(128, 128), batch_size=32,
    label_mode="categorical"
)

num_classes = len(train_ds.class_names)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, epochs=20, validation_data=test_ds)

test_photo = os.path.join(TEST_PATH, "images", "test.format.jpg")
if os.path.exists(test_photo):
    img = image.load_img(test_photo, target_size=(128, 128))
    img_array = np.expand_dims(image.img_to_array(img), axis=0)

    prediction = model.predict(img_array)
    result = np.argmax(prediction[0])
    print(f"Результат: {train_ds.class_names[result]} ({prediction[0][result]*100:.1f}%)")
