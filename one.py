from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model

from dnn.dnn import DNN

dnn = DNN()
dnn.load_data()
dnn.prepare_data()
dnn.build_model()

print(dnn.model.summary())
