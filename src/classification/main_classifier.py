import sys
import tensorflow as tf


def print_info():
    print('=' * 80)
    print('Python: ' + sys.version)
    print()

    print("# Tensorflow:")
    print('\t* TensorFlow version: {version}'.format(version=tf.__version__))
    print('\t* Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))
    print()

    cpus = tf.config.list_physical_devices('GPU')
    gpus = tf.config.list_physical_devices('GPU')

    print("# Devices:")
    print(f"\t* {len(cpus)} CPU(s):")
    print(f"\t* {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f'\t\t{i + 1}. {gpu}')
    print('=' * 80)


print_info()

from math import ceil
from art import *
from matplotlib import pyplot as plt
from IPython.core.display import display
from os import listdir
from os.path import isfile, join, basename
import numpy as np
import pandas as pd


def get_csvs_in_dir(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.csv')]


def plot_data(df, title='Dataset', n_cols=2, n_fullsize=1, fixed_bins=None, figsize=(12, 14)):
    cm = plt.cm.get_cmap('RdYlBu_r')

    plt.figure(figsize=figsize)
    plt.suptitle(f'{title} Dataset')

    nrows = ceil((len(df.columns) + 1) / 2)

    for ax_i, df_i in enumerate(df.columns):
        if ax_i < n_fullsize:
            plt.subplot(nrows, 1, ax_i + 1)
        else:
            plt.subplot(nrows, n_cols, ax_i + n_fullsize + 1)

        is_cat = df[df_i].apply(type).eq(str).all()
        n_unique = df[df_i].nunique()

        if fixed_bins is None:
            num_bins = n_unique if is_cat else int(n_unique / 2) if np.sqrt(n_unique) < 20 else int(np.sqrt(n_unique))
        else:
            num_bins = fixed_bins
        # print(f'{is_cat} - {n_unique} - {num_bins}')

        plt.title(f'{df_i} ({num_bins} bins - {"categorical" if is_cat else "continuous"})')

        n, bins, patches = plt.hist(df[df_i], lw=2, ec='white', bins=num_bins)
        col = (n - n.min()) / np.ptp(n)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

    plt.tight_layout()
    plt.show()


def read_csv(filename, print_plot=False):
    df = pd.read_csv(filename)
    if print_plot:
        name = basename(filename).split('.')[0].upper()
        print('=' * 110)
        tprint(name, font='big')

        plot_data(df, name)
        display(df)

    return df


def read_datasets(data_dir):
    # get dataset files

    cars_1_files = get_csvs_in_dir(data_dir)
    # peek at cars 1 datasets

    df_audi = read_csv(cars_1_files[0], print_plot=False)
    df_bmw = read_csv(cars_1_files[1], print_plot=False)
    df_ford = read_csv(cars_1_files[2], print_plot=False)
    df_hyundi = read_csv(cars_1_files[3], print_plot=False)
    df_merc = read_csv(cars_1_files[4], print_plot=False)
    df_toyota = read_csv(cars_1_files[5], print_plot=False)

    # add column for brand of car
    df_audi['brand'] = 'audi'
    df_bmw['brand'] = 'bmw'
    df_ford['brand'] = 'ford'
    df_hyundi['brand'] = 'hyundi'
    df_merc['brand'] = 'merc'
    df_toyota['brand'] = 'toyota'

    # concatenate all dataframes together
    df_cars = pd.concat([df_audi, df_bmw, df_ford, df_hyundi, df_merc, df_toyota])

    # change column order to something that allows us to split it easier later on
    df_cars = df_cars[['brand', 'model', 'transmission', 'fuelType',
                       'year', 'mileage', 'tax', 'mpg', 'engineSize',
                       'price']]
    print('=' * 110)
    print('=' * 110)
    tprint('CONCATENATED', font='big')
    # plot_data(df_cars, n_cols=2, n_fullsize=2)
    return df_cars


df_cars = read_datasets('../../datasets/car-listing-1')

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from tensorflow.keras.utils import to_categorical


def clean_encode_dataset(df_cars_1):

    # TODO decide if we want to predict year or just brand/model
    # cars_1_y = df_cars_1[['year','brand','model']].to_numpy()
    cars_1_y = df_cars_1[['brand','model']].to_numpy()  # TODO - uncomment if first try w/ too many labels gives dogwater accuracy
    cars_1_X = df_cars_1.copy().drop(['year','brand','model'], axis=1).to_numpy()

    # join y labels as single label
    cars_1_y = np.array(['_'.join(r.astype('str')).replace(' ', '') for r in cars_1_y]).reshape(-1, 1)
    cars_1_y = OrdinalEncoder().fit_transform(cars_1_y)

    # temporarily separate categorical cols from numerical
    num_cols = cars_1_X[:, 2:]
    cat_cols = cars_1_X[:, :2]

    print('=' * 60)
    print("# One-Hot-Encoding")
    # One-Hot Encode string values
    enc = OneHotEncoder(sparse=False)
    cat_cols_enc = enc.fit_transform(cat_cols)

    cars_1_X_enc = np.hstack((cat_cols_enc, num_cols)).astype(np.float32)
    tprint('One-Hot-Encoded', font='big')
    print(f'* Mean: {cars_1_X_enc.mean()}')
    print(f'* StDev: {cars_1_X_enc.std()}')
    print()
    print(cars_1_X_enc)

    print('=' * 80)

    scaler = StandardScaler()
    scaler.fit(cars_1_X_enc)
    cars_1_X_enc = scaler.transform(cars_1_X_enc)
    tprint('Standard Scaled', font='big')
    print(f'* Mean: {cars_1_X_enc.mean()}')
    print(f'* StDev: {cars_1_X_enc.std()}')
    print()
    print(cars_1_X_enc)

    return cars_1_X, cars_1_X_enc, cars_1_y

cars_1_X, cars_1_X_enc, cars_1_y = clean_encode_dataset(df_cars)

n_classes = len(np.unique(cars_1_y))

from sklearn.model_selection import train_test_split

print("# Train-Test Split")
cars_1_y = to_categorical(cars_1_y)
print(cars_1_y)
X_train, X_test, X_train_enc, X_test_enc, y_train, y_test = train_test_split(cars_1_X, cars_1_X_enc, cars_1_y,
                                                                             test_size=0.3, shuffle=True,
                                                                             random_state=69)

tprint('Training', font='big')
print(f"- X_train shape: {X_train.shape}")
print(f"- X_train_enc shape: {X_train_enc.shape}")
print(f"- y_train shape: {y_train.shape}")
# plot_data(pd.DataFrame(X_train), title="Training", n_cols=2, n_fullsize=2)
print()
tprint('Testing', font='big')
print(f"- X_test shape: {X_test.shape}")
print(f"- X_test_enc shape: {X_test_enc.shape}")
print(f"- y_test shape: {y_test.shape}")
# plot_data(pd.DataFrame(X_test), title="Testing", n_cols=2, n_fullsize=2)

tprint('Testing - Encoded', font='big')
# plot_data(pd.DataFrame(X_test_enc), title="Testing One-Hot-Encoded", n_cols=2, n_fullsize=0, fixed_bins=5,
#           figsize=(12, 150))


from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras import losses
from keras import optimizers


def model_builder(hp):
    model = Sequential()

    hp_kernel_initializer = hp.Choice('kernel_initializer',
                                      values=['random_normal', 'random_uniform', 'zeros', 'glorot_normal',
                                              'glorot_uniform'])

    hp_input_activation = hp.Choice('input_activation',
                                    values=['relu', 'sigmoid', 'tanh'])

    # Define INPUT layer
    model.add(Dense(X_train_enc.shape[1], input_dim=X_train_enc.shape[1],
                    activation=hp_input_activation,
                    kernel_initializer=hp_kernel_initializer,
                    name='layer_input'))

    # Define HIDDEN layers
    for i in range(hp.Int('layers',
                          min_value=1, max_value=10, step=1)):
        hp_hidden_units = hp.Int(f'hidden_{i}_units',
                                 min_value=32, max_value=512, step=32)
        hp_hidden_activation = hp.Choice(f'hidden_{i}_activation',
                                         values=['relu', 'sigmoid', 'tanh'])
        model.add(Dense(hp_hidden_units,
                        activation=hp_hidden_activation,
                        kernel_initializer=hp_kernel_initializer,
                        name=f'layer_hidden_{i}'))

    # Define OUTPUT layer
    model.add(Dense(n_classes, activation='softmax',
                    kernel_initializer=hp_kernel_initializer,
                    name='layer_output'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile model
    model.compile(loss=losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(
                      learning_rate=hp_learning_rate),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    # maybe try SparseTopKCategoricalAccuracy()

    # Print out model summary
    # print(model.summary())
    return model

import IPython
from keras import callbacks

# Define checkpoint callback for model saving
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
log_dir = 'tb-logs'

cb_checkpoint = callbacks.ModelCheckpoint(
    f'models/{checkpoint_name}', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
cb_early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, verbose=1, mode='auto')
cb_tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# defining a call that will clean out output at the end of every training epoch
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


import keras_tuner as kt

tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=500,
                     factor=3,
                     directory='models',
                     project_name='car_predict')

print('=' * 60)
print("# ", end='')
print(tuner.search_space_summary())
print('=' * 60)
print()


tuner_search = tuner.search(X_train_enc, y_train, epochs=500, validation_split=0.2,
                            callbacks=[cb_early_stopping, cb_tensorboard])
# overwrite=True

best_hps_arr = tuner.get_best_hyperparameters(num_trials=10)

for i, best_hps in enumerate(best_hps_arr):
    print('=' * 60)
    tprint(f"Model {i+1}")
    print(f"- Optimal Input Layer Activation: {best_hps['input_activation']}\n"
          f"- Optimal Kernel Initializer: {best_hps['kernel_initializer']}\n"
          f"- Optimal Learning Rate (Adam): {best_hps['learning_rate']}\n"
          f"- Optimal Number of Hidden Layers: {best_hps['layers']}")

    for i in range(best_hps['layers']):
        units = best_hps[f'hidden_{i}_units']
        activation = best_hps[f'hidden_{i}_activation']
        print(f"\t{i + 1}. Units: {units}\tActivation: {activation}")
    print()

models = []
for best_hps in best_hps_arr:
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train_enc, y_train, epochs=500, validation_split=0.2)
    models.append((model, history))

import os
import pickle

for i, (m, h) in enumerate(models):
    os.makedirs('best_models', exist_ok=True)
    # Save with Keras model save
    m.save(f'best_models/model_{i + 1}.hdf5')
    # Also save as pickle object because why not
    with open(f'best_models/model_{i + 1}.pickle', 'wb') as f1:
        pickle.dump(m, f1)
    # Save the model history
    with open(f'best_models/model_{i + 1}_history.pickle', 'wb') as f2:
        pickle.dump(h, f2)