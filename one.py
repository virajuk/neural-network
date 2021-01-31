from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Model

from dnn.dnn import DNN

dnn = DNN()

dnn.load_data()
dnn.prepare_data()

# # dnn.build_model()
# dnn.convolution_model()
#
# dnn.compile_model()
#
# print(dnn.model.summary())
#
# dnn.train_model()
# dnn.evaluate_model()
# dnn.save_model()

dnn.load_model()
# dnn.evaluate_model()

for i in range(17, 150):
    dnn.prediction(i)
