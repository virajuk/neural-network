import numpy as np
import cv2

from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model, load_model
from keras.optimizers import Adam


class DNN:

    def __init__(self):
        self.no_of_classes = 10
        self.batch_size = 32
        self.epochs = 25
        self.classes = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog'
            , 'frog', 'horse', 'ship', 'truck'])

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
        x = Dense(units=300, activation='relu')(x)

        # layer 2
        x = Dense(units=250, activation='relu')(x)

        # layer 3
        x = Dense(units=200, activation='relu')(x)

        # layer 4
        x = Dense(units=150, activation='relu')(x)

        # layer 5
        x = Dense(units=100, activation='relu')(x)

        output_layer = Dense(units=10, activation='softmax')(x)

        self.model = Model(input_layer, output_layer)

    def compile_model(self):

        opt = Adam(lr=0.0005)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train_model(self):

        self.model.fit(self.x_train
                       , self.y_train
                       , self.batch_size
                       , self.epochs
                       , shuffle=True
                       , verbose=2)

    def evaluate_model(self):

        # print(self.model.metrics_names)
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print(scores)

    def save_model(self):

        self.model.save('model/my_ass.pb')

    def load_model(self):

        self.model = load_model('model/my_ass.pb')

    def prediction(self, idx):

        preds = self.model.predict(self.x_test)
        preds_single = self.classes[np.argmax(preds, axis=-1)]
        actual_single = self.classes[np.argmax(self.y_test, axis=-1)]

        image_name = f"actual : {actual_single[idx]}, pred : { preds_single[idx]}"

        cv2.imshow(image_name, self.x_test[idx])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
