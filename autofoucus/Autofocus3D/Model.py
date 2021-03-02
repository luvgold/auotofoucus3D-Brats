from Autofocus3D.autofocus3D import Autofocus3D
from Autofocus3D.ResBlock import BasicBlock
from Autofocus3D.autofocus_single import Autofocus_single
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


def build_model():
    # Dilation rates, here 4 parallel conv applications
    dilations = [2, 6, 10, 14]
    # channels = [num_input - 1, 30, 30, 40, 40, 40, 40, 50, 50, num_classes]
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(30, kernel_size=3, dilation_rate=[1, 1, 1], padding='SAME'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv3D(30, kernel_size=3, dilation_rate=[1, 1, 1], padding='SAME'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv3D(40, kernel_size=3, dilation_rate=[1, 1, 1], padding='SAME'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv3D(40, kernel_size=3, dilation_rate=[1, 1, 1], padding='SAME'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Activation("relu"),

        BasicBlock(40,40,40),
        # BasicBlock(40,40, 40),
        # BasicBlock(40,40, 40),
        # BasicBlock(40,40, 40),

        # Autofocus_single(40, 50, 50, dilations, 4),

        Autofocus3D(dilations,
                    filters=50,
                    kernel_size=(3, 3, 3),
                    activation='relu',
                    attention_activation=tf.nn.relu,
                    attention_filters=25,
                    attention_kernel_size=3,
                    use_bn=True,
                    use_bias=True),
        tf.keras.layers.Conv3D(3, 1, activation="relu")
        # etc....
    ])
    return model
def build_graph(model):
    # Build model by passing random data...

    writer = tf.summary.FileWriter(logdir="./")
    with tf.Session() as s:
        writer.add_graph(s.graph)
        writer.flush()


