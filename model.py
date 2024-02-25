import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1, activation='sigmoid'))  # lub inną odpowiednią funkcję aktywacji, np. 'relu' dla wartości dodatnich
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # dostosuj funkcję straty i metryki do Twojego przypadku
    return model