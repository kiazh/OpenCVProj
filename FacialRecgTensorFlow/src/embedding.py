"""Embedding network — notebook cells 17-18."""
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model

from .config import EMBEDDING_DIM, IMAGE_SIZE


def make_embedding(
    input_shape=(*IMAGE_SIZE, 3),
    embedding_dim=EMBEDDING_DIM,
):
    inp = Input(shape=input_shape, name="input_image")

    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D((2, 2), padding="same")(c1)

    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D((2, 2), padding="same")(c2)

    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D((2, 2), padding="same")(c3)

    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(embedding_dim, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=d1, name="embedding")
