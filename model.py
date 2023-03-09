import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Model
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Model


def get_aec_model():
    irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax", )
    conv = irv2.layers[-20].output
    conv = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)
    output = Flatten()(conv)
    output = Dense(4, activation='softmax')(output)
    conv_model = Model(inputs=irv2.input, outputs=output)

    return conv_model


def get_prediction(img: Image.Image, model: tf.keras.Model):

    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    normalized_image = np.vstack([img_batch]) / 255.0
    probabilities = model.predict(normalized_image)

    return probabilities
