from __future__ import absolute_import, division, print_function, unicode_literals

import math
import tensorflow as tf

# Flip image
def flip_image(image, label):
    image = tf.image.flip_left_right(image)
    return image, label


# Rotate image by 3 degrees
def rotate_image_1(image, label):
    image = tf.contrib.image.rotate(image, math.radians(3))
    return image, label


# Rotate image by -3 degrees
def rotate_image_2(image, label):
    image = tf.contrib.image.rotate(image, math.radians(-3))
    return image, label


# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
def scale_image(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


# Create neural network model
def build_model():
    # Declare model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same', input_shape=(32, 32, 3)), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile it
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.9, decay=1e-6),
      metrics=['accuracy']
    )

    # Output summary and return
    model.summary()
    return model


# Create neural network model
def build_model_simplified():
    # Declare model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(12, 3, activation='relu', padding='same', input_shape=(32, 32, 3)), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(24, 3, activation='relu', padding='same'), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile it
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.9, decay=1e-6),
        metrics=['accuracy']
    )

    # Output summary and return
    model.summary()
    return model


# Augment dataset
def augment_dataset(dataset_train_raw):
    dataset_train_flipped = dataset_train_raw.map(flip_image)
    dataset_train_rotated_1 = dataset_train_raw.map(rotate_image_1)
    dataset_train_rotated_2 = dataset_train_raw.map(rotate_image_2)
    return dataset_train_raw.concatenate(dataset_train_flipped).concatenate(dataset_train_rotated_1).concatenate(dataset_train_rotated_2)