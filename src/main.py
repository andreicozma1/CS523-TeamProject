# import libraries
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
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# get dataset files

cars_listing_1_dir = '../datasets/car-listing-1'
cars_listing_2_dir = '../datasets/car-listing-2'


def get_csvs_in_dir(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.csv')]


cars_1_files = get_csvs_in_dir(cars_listing_1_dir)
cars_2_files = get_csvs_in_dir(cars_listing_2_dir)


# peek at cars 1 datasets
df_audi = pd.read_csv(cars_1_files[0])
print(df_audi.head())

df_bmw = pd.read_csv(cars_1_files[1])
print(df_bmw.head())

df_ford = pd.read_csv(cars_1_files[2])
print(df_ford.head())

df_hyundi = pd.read_csv(cars_1_files[3])
print(df_hyundi.head())

df_merc = pd.read_csv(cars_1_files[4])
print(df_merc.head())

df_toyota = pd.read_csv(cars_1_files[5])
print(df_toyota.head())


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
cars_1_y = df_cars_1.pop('price').to_numpy()
cars_1_X = df_cars_1.to_numpy()


# temporarily separate categorical cols from numerical
num_cols = cars_1_X[:, 4:]
cat_cols = cars_1_X[:, :4]

# One-Hot Encode string values
enc = OneHotEncoder(sparse=False)
cat_cols_enc = enc.fit_transform(cat_cols)

cars_1_X_enc = np.hstack((cat_cols_enc, num_cols)).astype(np.float32)
print(cars_1_X_enc.shape)
print(cars_1_X_enc[:10])


X_train, X_test, y_train, y_test = train_test_split(
    cars_1_X_enc, cars_1_y, test_size=0.2, shuffle=True)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_test shape: {y_test.shape}")


# TODO - try other loss functions and determine best to use


def BaselineModel(optimizer='adam', loss='mean_squared_error',
                  activation='relu', output_activation='linear',
                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                  input_neurons=150, hidden_neurons=200, num_hidden_layers=2):

    model = Sequential()

    # Define INPUT layer
    model.add(Dense(input_neurons, input_dim=X_train.shape[1],
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    name='layer_input'))

    for i in range(num_hidden_layers):
        model.add(Dense(hidden_neurons, activation=activation,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        name=f'layer_hidden_{i}'))

    # Define OUTPUT layer
    model.add(Dense(1, activation=output_activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    name='layer_output'))

    # Compile model
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Print out model summary
    print(model.summary())
    return model


# Define checkpoint callback for model saving
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
log_dir = 'tensorboard-logs'

checkpoint = ModelCheckpoint(
    f'models/{checkpoint_name}', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks_list = [checkpoint, early_stopping, tensorboard]


# Run with default parameters

coarse_grid = [
    {
        'batch_size': [10, 50, 100, 200, 400],
        'epochs': [25, 50, 100],
        'optimizer': ['Adam'],
        'activation': ['relu'],
        'num_hidden_layers': [1, 2, 5],
        'hidden_neurons': [50, 100, 200, 500]
    }
]

cross_val = 5
scores = ['precision', 'recall']
result = {}
for score in scores:
    print('-' * 50)
    print(
        f"# Tuning hyper-parameters for {score} with {cross_val}-fold cross-validation")
    print()

    # Employ GridSearch using the cross_val variable on the param grid provided
    clf = GridSearchCV(
        KerasRegressor(build_fn=BaselineModel,
                       batch_size=100, epochs=100, verbose=2),
        coarse_grid,
        scoring='%s_macro' % score,
        # cv=cross_val,
        n_jobs=2,
        verbose=2
    )
    # Fit the model on the training labels and outputs

    clf.fit(X_train, y_train,
            shuffle=True,
            workers=8, use_multiprocessing=True)

    # clf.fit(X_train, y_train,
    #         callbacks=[early_stopping],
    #         verbose=2, shuffle=True,
    #         workers=2, use_multiprocessing=True)

    print("# Best parameters set found on development set:")
    print(f'\t{clf.best_params_}')
    print()
    result[score] = clf.best_params_
    print("# Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("\t - %0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("# Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
