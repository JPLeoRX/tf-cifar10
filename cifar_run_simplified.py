from __future__ import absolute_import, division, print_function, unicode_literals
from cifar import *

import tensorflow as tf
import tensorflow_datasets as tfds

# Define distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_OF_WORKERS = strategy.num_replicas_in_sync
print("{} replicas in distribution".format(NUM_OF_WORKERS))

# Determine datasets buffer/batch sizes
BUFFER_SIZE = 20000
BATCH_SIZE_PER_REPLICA = 128
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
dataset_train_augmented = augment_dataset(dataset_train_raw)
dataset_train = dataset_train_augmented.map(scale_image).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).with_options(options)
dataset_test = dataset_test_raw.map(scale_image).batch(BATCH_SIZE).with_options(options)

# Build and train the model as multi worker
with strategy.scope():
    model = build_model_simplified()
model.fit(x=dataset_train, epochs=60)

# Show model summary, and evaluate it
model.summary()
eval_loss, eval_acc = model.evaluate(x=dataset_test)
print("\nEval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))

# Save the model
model.save("model_simplified.h5")