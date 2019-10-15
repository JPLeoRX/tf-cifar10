from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds


# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
def scale_image(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


# Create neural network model
def build_model():
    # Declare model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(100, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Conv2D(200, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(200, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.40),

        tf.keras.layers.Conv2D(300, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(300, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.45),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile it
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      metrics=['accuracy']
    )

    # Output summary and return
    model.summary()
    return model


# Define distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_OF_WORKERS = strategy.num_replicas_in_sync
print("{} replicas in distribution".format(NUM_OF_WORKERS))

# Determine datasets buffer/batch sizes
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_OF_WORKERS
print("{} batch size".format(BATCH_SIZE))

# Define and load datasets
datasets, info = tfds.load(name='cifar10', with_info=True, as_supervised=True)
NUM_OF_TRAIN_SAMPLES = info.splits['train'].num_examples
NUM_OF_TEST_SAMPLES = info.splits['test'].num_examples
print("{} samples in training dataset, {} samples in testing dataset".format(NUM_OF_TRAIN_SAMPLES, NUM_OF_TEST_SAMPLES))
dataset_train_raw = datasets['train']
dataset_test_raw = datasets['test']

# Prepare training/testing dataset
options = tf.data.Options()
options.experimental_distribute.auto_shard = False
dataset_train = dataset_train_raw.map(scale_image).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).with_options(options)
dataset_test = dataset_test_raw.map(scale_image).batch(BATCH_SIZE).with_options(options)

# Build and train the model as multi worker
with strategy.scope():
    model = build_model()
model.fit(x=dataset_train, epochs=24)

# Show model summary, and evaluate it
model.summary()
eval_loss, eval_acc = model.evaluate(x=dataset_test)
print("\nEval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))