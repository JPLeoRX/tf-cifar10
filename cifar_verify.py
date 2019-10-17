from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
def scale_image(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

# Define distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_OF_WORKERS = strategy.num_replicas_in_sync
print("{} replicas in distribution".format(NUM_OF_WORKERS))

# Determine datasets buffer/batch sizes
BATCH_SIZE_PER_REPLICA = 128
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_OF_WORKERS
print("{} batch size".format(BATCH_SIZE))

# Define and load datasets
datasets, info = tfds.load(name='cifar10', with_info=True, as_supervised=True)
NUM_OF_TEST_SAMPLES = info.splits['test'].num_examples
print("{} samples in testing dataset".format(NUM_OF_TEST_SAMPLES))
dataset_test_raw = datasets['test']

# Prepare training/testing dataset
options = tf.data.Options()
options.experimental_distribute.auto_shard = False
dataset_test = dataset_test_raw.map(scale_image).batch(BATCH_SIZE).with_options(options)

# Load model
model = tf.keras.models.load_model("model_9147.h5")

# Show model summary, and evaluate it
model.summary()
eval_loss, eval_acc = model.evaluate(x=dataset_test)
print("\nEval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))