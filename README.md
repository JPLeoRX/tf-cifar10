# Tensorflow CIFAR10

Neural network model for common CIFAR10 benchmark problem in Machine Learning

[Tensorflow Datasets - CIFAR10](https://www.tensorflow.org/datasets/catalog/cifar10)

[About CIFAR10 (University of Toronto - Computer Science)](https://www.cs.toronto.edu/%7Ekriz/cifar.html)

[Benchmarked Results of Other Models](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)

# Results

The obtained model achieves 91.49% accuracy, which can be verified by ```cifar_verify.py``` script

# Model Topology

The resulting model is a Deep Convolutional Neural Network. 
It has a total of 12 convolution layers and 2 regular layers.
Convolutional layers are split in 4 groups by 3 layers in each group, with batch normalization after each layer. 
At the end of each group there's a max pooling layer and a dropout layer.

# Dataset Augmentation

To improve training the dataset was slightly augmented. 
A horizontally flipped copy of each image was added, as well as two rotated (-3 and +3 degrees) copies of each image were added. 
Resulting training dataset reached 200 000 images.

# Hardware
Training of this model was performed on two Nvidia GeForce GTX 1070 cards, with Intel Core i5 8600K, and 16GB RAM.