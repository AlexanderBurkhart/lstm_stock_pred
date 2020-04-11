from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras import optimizers

def lstm_model(batch_input_shape):
    m = Sequential()
    m.add(LSTM(100, batch_input_shape=batch_input_shape,
                dropout=0.0, recurrent_dropout=0.0, stateful=-True,
                kernel_initializer='random_uniform'))
    m.add(Dropout(0.5))
    m.add(Dense(20,activation='relu'))
    m.add(Dense(1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=0.001)
    m.compile(loss='mean_squared_error', optimizer=optimizer)
    return m
