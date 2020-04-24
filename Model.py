import keras

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers import LSTM

def get_model():
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_trainshape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model