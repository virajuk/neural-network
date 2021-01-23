import numpy as np

from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model
from keras.optimizers import Adam


class DNN:

    def __init__(self):
        self.no_of_classes = 10
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

    def prepare_data(self):
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        self.y_train = to_categorical(self.y_train, self.no_of_classes)
        self.y_test = to_categorical(self.y_test, self.no_of_classes)

    def build_model(self, input_shape=(32, 32, 3)):

        input_layer = Input(shape=input_shape, name='input_layer')

        x = Flatten(name='flatten_layer')(input_layer)

        # layer 1
        x = Dense(units=200, activation='relu')(x)

        # layer 2
        x = Dense(units=150, activation='relu')(x)

        output_layer = Dense(units=10, activation='softmax')(x)

        self.model = Model(input_layer, output_layer)
