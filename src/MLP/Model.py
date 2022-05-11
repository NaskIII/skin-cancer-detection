from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model() -> Sequential:
    model: Sequential = Sequential()
    model.add(Dense(16, input_dim=9, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
