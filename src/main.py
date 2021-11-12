# import libraries
from keras import callbacks
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras import metrics
from keras import losses
from keras import optimizers


def print_info():
    print('=' * 60)

    print('- TensorFlow version: {version}'.format(version=tf.__version__))
    print('- Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))

    gpus = tf.config.list_physical_devices('GPU')
    print(f"- Num GPUs: {len(gpus)}")
    print(f'- GPUs: {gpus}')
    print('=' * 60)
    print()


def get_csvs_in_dir(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.csv')]


def read_dataset():
    # get dataset files

    cars_listing_1_dir = '../datasets/car-listing-1'
    cars_listing_2_dir = '../datasets/car-listing-2'
    cars_1_files = get_csvs_in_dir(cars_listing_1_dir)
    cars_2_files = get_csvs_in_dir(cars_listing_2_dir)

    # peek at cars 1 datasets
    print('=' * 60)

    print("# AUDI")
    df_audi = pd.read_csv(cars_1_files[0])
    print(df_audi.head())

    print('=' * 60)

    print("# BMW")
    df_bmw = pd.read_csv(cars_1_files[1])
    print(df_bmw.head())

    print('=' * 60)

    print("# FORD")
    df_ford = pd.read_csv(cars_1_files[2])
    print(df_ford.head())

    print('=' * 60)

    print("# HYUNDAI")
    df_hyundi = pd.read_csv(cars_1_files[3])
    print(df_hyundi.head())

    print('=' * 60)

    print("# MERCEDES")
    df_merc = pd.read_csv(cars_1_files[4])
    print(df_merc.head())

    print('=' * 60)

    print("# TOYOTA")
    df_toyota = pd.read_csv(cars_1_files[5])
    print(df_toyota.head())

    print('=' * 60)

    # add column for brand of car
    df_audi['brand'] = 'audi'
    df_bmw['brand'] = 'bmw'
    df_ford['brand'] = 'ford'
    df_hyundi['brand'] = 'hyundi'
    df_merc['brand'] = 'merc'
    df_toyota['brand'] = 'toyota'

    # concatenate all dataframes together
    df_cars_1 = pd.concat(
        [df_audi, df_bmw, df_ford, df_hyundi, df_merc, df_toyota])

    # change column order to something that allows us to split it easier later on
    df_cars_1 = df_cars_1[['brand', 'model', 'transmission', 'fuelType',
                           'year', 'mileage', 'tax', 'mpg', 'engineSize', 'price']]

    df_cars_1


def clean_encode_dataset(df_cars_1):
    cars_1_y = df_cars_1.pop('price').to_numpy()

    cars_1_X = df_cars_1.to_numpy()

    # temporarily separate categorical cols from numerical
    num_cols = cars_1_X[:, 4:]
    cat_cols = cars_1_X[:, :4]

    print('=' * 60)
    print("# One-Hot-Encoding")
    # One-Hot Encode string values
    enc = OneHotEncoder(sparse=False)
    cat_cols_enc = enc.fit_transform(cat_cols)

    cars_1_X_enc = np.hstack((cat_cols_enc, num_cols)).astype(np.float32)
    print(cars_1_X_enc.shape)
    print(cars_1_X_enc[:10])

    return cars_1_X_enc, cars_1_y


df_cars_1 = read_dataset()

cars_1_X_enc, cars_1_y = clean_encode_dataset(df_cars_1)


print('=' * 60)


print('=' * 60)
print("# Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    cars_1_X_enc, cars_1_y, test_size=0.2, shuffle=True)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_test shape: {y_test.shape}")

print('=' * 60)


def model_builder(hp):

    model = Sequential()

    hp_kernel_initializer = hp.Choice(
        'kernel_initializer', values=['random_normal', 'random_uniform', 'zeros', 'glorot_normal', 'glorot_uniform'])

    hp_input_activation = hp.Choice('input_activation',
                                    values=['relu', 'sigmoid', 'tanh'])

    # Define INPUT layer
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation=hp_input_activation,
                    kernel_initializer=hp_kernel_initializer,
                    name='layer_input'))

    for i in range(hp.Int('layers', 1, 6)):
        hp_hidden_units = hp.Int(
            f'hidden_{i}_units', min_value=32, max_value=512, step=32)
        hp_hidden_activation = hp.Choice(f'hidden_{i}_activation',
                                         values=['relu', 'sigmoid', 'tanh'])
        model.add(Dense(hp_hidden_units, activation=hp_hidden_activation,
                        kernel_initializer=hp_kernel_initializer,
                        name=f'layer_hidden_{i}'))

    # Define OUTPUT layer
    model.add(Dense(1, activation='linear',
                    kernel_initializer=hp_kernel_initializer,
                    name='layer_output'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile model
    model.compile(loss=losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=hp_learning_rate),
                  metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError(), metrics.RootMeanSquaredError()])

    # Print out model summary
    # print(model.summary())
    return model


print('=' * 60)

# Define checkpoint callback for model saving
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
log_dir = 'tb-logs'


cb_checkpoint = ModelCheckpoint(
    f'models/{checkpoint_name}', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
cb_early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, verbose=1,  mode='auto')
cb_tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)


tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=500,
                     factor=3,
                     directory='models',
                     project_name='car_prices',
                     overwrite=True)

tuner.search(X_train, y_train, epochs=500, validation_split=0.2,
             callbacks=[cb_early_stopping, cb_tensorboard])


print('=' * 60)
print("# ", end='')
print(tuner.search_space_summary())


print('=' * 60)
print("# ", end='')
summary = tuner.results_summary(num_trials=1)
print(summary)


print('=' * 60)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f'Best Hyperparams: {best_hps}')

"""
# Search space summary
Default search space size: 16
kernel_initializer (Choice)
{'default': 'random_normal', 'conditions': [], 'values': ['random_normal',
    'random_uniform', 'zeros', 'glorot_normal', 'glorot_uniform'], 'ordered': False}
input_activation (Choice)
{'default': 'relu', 'conditions': [], 'values': [
    'relu', 'sigmoid', 'tanh'], 'ordered': False}
layers (Int)
{'default': None, 'conditions': [], 'min_value': 1,
    'max_value': 6, 'step': 1, 'sampling': None}
hidden_0_units (Int)
{'default': None, 'conditions': [], 'min_value': 32,
    'max_value': 512, 'step': 32, 'sampling': None}
hidden_0_activation (Choice)
{'default': 'relu', 'conditions': [], 'values': [
    'relu', 'sigmoid', 'tanh'], 'ordered': False}
learning_rate (Choice)
{'default': 0.01, 'conditions': [], 'values': [
    0.01, 0.001, 0.0001], 'ordered': True}
hidden_1_units (Int)
{'default': None, 'conditions': [], 'min_value': 32,
    'max_value': 512, 'step': 32, 'sampling': None}
hidden_1_activation (Choice)
{'default': 'relu', 'conditions': [], 'values': [
    'relu', 'sigmoid', 'tanh'], 'ordered': False}
hidden_2_units (Int)
{'default': None, 'conditions': [], 'min_value': 32,
    'max_value': 512, 'step': 32, 'sampling': None}
hidden_2_activation (Choice)
{'default': 'relu', 'conditions': [], 'values': [
    'relu', 'sigmoid', 'tanh'], 'ordered': False}
hidden_3_units (Int)
{'default': None, 'conditions': [], 'min_value': 32,
    'max_value': 512, 'step': 32, 'sampling': None}
hidden_3_activation (Choice)
{'default': 'relu', 'conditions': [], 'values': [
    'relu', 'sigmoid', 'tanh'], 'ordered': False}
hidden_4_units (Int)
{'default': None, 'conditions': [], 'min_value': 32,
    'max_value': 512, 'step': 32, 'sampling': None}
hidden_4_activation (Choice)
{'default': 'relu', 'conditions': [], 'values': [
    'relu', 'sigmoid', 'tanh'], 'ordered': False}
hidden_5_units (Int)
{'default': None, 'conditions': [], 'min_value': 32,
    'max_value': 512, 'step': 32, 'sampling': None}
hidden_5_activation (Choice)
{'default': 'relu', 'conditions': [], 'values': [
    'relu', 'sigmoid', 'tanh'], 'ordered': False}
None
============================================================
# Results summary
Results in models/car_prices
Showing 1 best trials
Objective(name='val_loss', direction='min')
Trial summary
Hyperparameters:
kernel_initializer: glorot_normal
input_activation: relu
layers: 2
hidden_0_units: 128
hidden_0_activation: relu
learning_rate: 0.01
hidden_1_units: 224
hidden_1_activation: relu
hidden_2_units: 288
hidden_2_activation: tanh
hidden_3_units: 480
hidden_3_activation: relu
hidden_4_units: 320
hidden_4_activation: sigmoid
hidden_5_units: 480
hidden_5_activation: relu
tuner/epochs: 34
tuner/initial_epoch: 12
tuner/bracket: 3
tuner/round: 2
tuner/trial_id: 9bf58eee151b6212ad5a66d0d53bb522
Score: 17684080.0
None
"""
