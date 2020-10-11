import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16


def base_network():
    model = VGG16(include_top=True, weights=None)
    dense = Dense(128)(model.layers[-4].output)
    norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense)
    model = Model(inputs=[model.input], outputs=[norm2])
    return model
