

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import tensorflow as tf


class InsertNet:

    @staticmethod
    def simple_model(inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        # x = tf.keras.layers.Dense(512, activation="relu")(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        # outputs = tf.keras.layers.Dense(len(config.CLASSES), activation=last_activation)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        # no activation here, will get logits as predictions
        return outputs

    @staticmethod
    def complex_model(img_size, channels , num_classes, last_activation):

        model = Sequential()
        inputShape = (img_size[0], img_size[1], channels)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape =  (channels, img_size[0], img_size[1])
            chanDim = 1

        # CONV => RELU => POOL
        model.add(SeparableConv2D(32, (3, 3), padding="same",input_shape= inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 2
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 3
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # sigmoid classifier
        model.add(Dense(1))
        model.add(Activation(last_activation))

        return model