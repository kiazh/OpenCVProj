"""Siamese assembly + persistence — notebook cells 23-24, 43, 45."""
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model

from .config import IMAGE_SIZE
from .distance import L1Dist
from .embedding import make_embedding


def make_siamese_model(embedding=None, input_shape=(*IMAGE_SIZE, 3)):
    embedding = embedding or make_embedding()

    input_image = Input(name="input_img", shape=input_shape)
    validation_image = Input(name="validation_img", shape=input_shape)

    siamese_layer = L1Dist()
    siamese_layer._name = "distance"
    distances = siamese_layer([embedding(input_image), embedding(validation_image)])

    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(
        inputs=[input_image, validation_image],
        outputs=classifier,
        name="SiameseNetwork",
    )


def save_model(model, path):
    model.save(path)


def load_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "L1Dist": L1Dist,
            "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
        },
    )
