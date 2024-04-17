#for now it's only a draft

"""
Architecture of my custom nn:
- input layer - information about teams for specific date on which a match against them is played
- analysis (?) layer - information about h2h games from the past
- hidden layers
- output layer - prediction of result, amount of goals scored, corner kicks, yellow/red cards and odds for the game (based on odds from training set)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class Predictor:
    def __init__(self, history, stats):
        self.history = history
        self.stats = stats
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.history_length, self.features_per_match)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5, activation='linear'))  # 5 outputów: wynik, liczba goli, liczba rzutów rożnych, liczba kartek, kurs spotkania
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_new_data):
        return self.model.predict(X_new_data)
        