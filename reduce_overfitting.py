# model only memorized training images instead of actually learning to recognize cats and dogs. It happen when it perform great on
# training data but struggles on new images -> that is calles overfitting.

# We can fix overfiting by using dropout, data Augmentation, and early Stoppping.


# Data Augmentation -> flips, zoom, rotate - basically give model more variety to learn from.

import tensorflow as tf

def get_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])


# Drop out -> rondomly turns off neuron during training - force model to not rely on memorization