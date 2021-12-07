from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras import losses
from keras import optimizers


def model_builder(hp):
    model = Sequential()

    hp_kernel_initializer = hp.Choice('kernel_initializer',
                                      values=['glorot_normal',
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
                          min_value=5, max_value=15, step=1)):
        hp_hidden_units = hp.Int(f'hidden_{i}_units',
                                 min_value=256, max_value=2048, step=128)
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

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

    # Compile model
    model.compile(loss=losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=hp_learning_rate),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    # maybe try SparseTopKCategoricalAccuracy()

    # Print out model summary
    # print(model.summary())
    return model

