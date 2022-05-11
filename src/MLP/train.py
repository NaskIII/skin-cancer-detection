import os

import time
from datetime import datetime

import pandas as pd
from numpy import ndarray
from pandas import DataFrame

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, Nadam, SGD

from sklearn.model_selection import train_test_split

from Model import create_model


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path: str, random_seed: int, split: float = 0.2):
    df_x: DataFrame = pd.read_csv(path)
    df_y: DataFrame = DataFrame(data=df_x.pop("diagnose"))

    df_x: ndarray = df_x.values
    df_y: ndarray = df_y.values

    test_size: int = int(df_x.shape[0] * split)

    data_train_x, data_valid_x = train_test_split(df_x, test_size=test_size, random_state=random_seed)
    data_train_y, data_valid_y = train_test_split(df_y, test_size=test_size, random_state=random_seed)

    data_train_x, data_test_x = train_test_split(data_train_x, test_size=test_size, random_state=random_seed)
    data_train_y, data_test_y = train_test_split(data_train_y, test_size=test_size, random_state=random_seed)

    return (data_train_x, data_train_y), (data_valid_x, data_valid_y), (data_test_x, data_test_y)


"""
def load_data(path: str, samples):
    dataset_train = np.loadtxt(path, delimiter=",")
    indices = np.random.choice(dataset_train.shape[0], samples, replace=False)
    dataset_test = dataset_train[indices]
    dataset_train = np.delete(dataset_train, indices, axis=0)
    return (dataset_train, dataset_test)"""


""" Seeds """
seed: int = int(time.time())
day: str = datetime.today().strftime('%Y-%m-%d %H:%M:%S').replace(':', '')

np.random.seed(seed)
tf.random.set_seed(seed)

"""Dataset: 60/20/20"""
dataset: str = r"../Data/normalized-golden-segmented-descriptors.csv"
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset, 2324)

create_dir(r"../logs/seeds")

with open(f"../logs/seeds/{day}.txt", 'a', encoding="utf-8") as file:
    file.write(f"The seed that was used in training {day} was: {seed}")

create_dir(rf"../files/{day}")

""" Hyperparameters """
batch_size = 50
num_epoch = 200
model_path = fr"../files/{day}/model.h5"
csv_path = fr"../files/{day}/data.csv"

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")
print(f"Test: {len(test_x)} - {len(test_y)}")

""" Create New Model """
model = create_model()
metrics = [AUC(curve="ROC"), Recall(), Precision()]
model.compile(loss="binary_crossentropy", optimizer=RMSprop(), metrics=metrics)
model.summary()

callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, min_lr=1e-9, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='auto')
]

model.fit(
    x=train_x,
    y=train_y,
    epochs=num_epoch,
    validation_data=(valid_x, valid_y),
    callbacks=callbacks
)

model = tf.keras.models.load_model(fr"../files/{day}/model.h5")

predictions = (model.predict(test_x) > 0.5).astype("int32")

correct = 0
incorrect = 0
total_spec = 0
total_sensi = 0
TP = 0
FP = 0
TN = 0
FN = 0

for i in test_y:
    if i == 0:
        total_spec += 1
    else:
        total_sensi += 1

for i in range(len(test_x)):
    if predictions[i] == test_y[i]:
        correct += 1
        if predictions[i] == 0:
            TN += 1
        elif predictions[i] == 1:
            TP += 1
    else:
        incorrect += 1
        if predictions[i] == 0:
            FN += 1
        elif predictions[i] == 1:
            FP += 1

print('''
Corretas: %s; 
Incorretas: %s; 
Especificidade: %s; 
Sensibilidade: %s;
    ''' %
      (correct, incorrect, TN, TP))

print('Total Especificidade: %s; Total Sensibilidade: %s;' % (total_spec, total_sensi))

print('''
Porcentagem Especificidade: %s; 
Porcentagem Sensibilidade: %s; 
Porcentagem Acurácia: %s;
    ''' %
      (TN / total_spec * 100, TP / total_sensi * 100, correct / len(test_x) * 100))

print('Especificidade: %s' % ((TN / (TN + FP)) * 100))
print('Sensibilidade: %s' % ((TP / (TP + FN)) * 100))
print('Precisão: %s' % ((TP / (TP + FP)) * 100))
print('Acurácia: %s' % (((TP + TN) / (TP + TN + FP + FN)) * 100))

with open(f"../files/{day}/eval.txt", 'a', encoding="utf-8") as file:
    file.write(f'''
Corretas: {correct}; 
Incorretas: {incorrect}; 
Especificidade: {TN}; 
Sensibilidade: {TP};

Total Especificidade: {total_spec}; Total Sensibilidade: {total_sensi};

Porcentagem Especificidade: {TN / total_spec * 100}; 
Porcentagem Sensibilidade: {TP / total_sensi * 100}; 
Porcentagem Acurácia: {correct / len(test_x) * 100};
''')
