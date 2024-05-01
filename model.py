#for now it's only a draft

"""
Architecture of my custom nn:
- input layer - information about teams for specific date on which a match against them is played
- analysis hidden layer - information about h2h games from the past
- hidden layers
- output layer - prediction of result, amount of goals scored, corner kicks, yellow/red cards and odds for the game (based on odds from training set)
"""

import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, concatenate, LSTM, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class FootballPredictor:
    def __init__(self):
        self.model = self._build_model()
        
        #instances for data preprocessing inside a model (extracting h2h history)
        self.encoder = OneHotEncoder()
        self.standard_scaler = StandardScaler()
        
        self.model_params = {}
        self.training_params = {}
                
    def build_model(self, input_shape):
        # Wprowadzenie danych wejściowych
        input_layer = Input(shape=(input_shape,), name='input_layer')

        # Wspólna baza sieci
        x = Dense(128, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)

        # Przewidywanie rezultatu meczu (np. wygrana gospodarzy, remis, wygrana gości)
        result_output = Dense(3, activation='softmax', name='result_output')(x)

        # Przewidywanie liczby strzelonych goli (regresja)
        goals_output = Dense(1, activation='linear', name='goals_output')(x)

        # Przewidywanie liczby rzutów rożnych (regresja)
        corners_output = Dense(1, activation='linear', name='corners_output')(x)

        # Przewidywanie liczby kartek (regresja dla żółtych i czerwonych kartek)
        yellow_cards_output = Dense(1, activation='linear', name='yellow_cards_output')(x)
        red_cards_output = Dense(1, activation='linear', name='red_cards_output')(x)

        # Przewidywanie kursów na mecz (regresja)
        odds_home_win_output = Dense(1, activation='linear', name='odds_home_win_output')(x)
        odds_draw_output = Dense(1, activation='linear', name='odds_draw_output')(x)
        odds_away_win_output = Dense(1, activation='linear', name='odds_away_win_output')(x)

        # Zbudowanie modelu
        model = Model(
            inputs=input_layer,
            outputs=[result_output, goals_output, corners_output, yellow_cards_output, red_cards_output, odds_home_win_output, odds_draw_output, odds_away_win_output]
        )

        # Kompilacja modelu
        model.compile(optimizer='adam',
                      loss={'result_output': 'categorical_crossentropy', 'goals_output': 'mse', 'corners_output': 'mse',
                            'yellow_cards_output': 'mse', 'red_cards_output': 'mse', 'odds_home_win_output': 'mse',
                            'odds_draw_output': 'mse', 'odds_away_win_output': 'mse'},
                      metrics={'result_output': 'accuracy', 'goals_output': 'mse', 'corners_output': 'mse',
                               'yellow_cards_output': 'mse', 'red_cards_output': 'mse', 'odds_home_win_output': 'mse',
                               'odds_draw_output': 'mse', 'odds_away_win_output': 'mse'})
        return model
    

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_new_data):
        return self.model.predict(X_new_data)