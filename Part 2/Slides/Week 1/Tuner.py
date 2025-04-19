# IN ANACONDA: 'activate tensorflow' to get into the right environment

import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import keras_tuner

(trainval_images, trainval_labels), (test_images, test_labels) = mnist.load_data()

# Reshape data
trainval_images = trainval_images.reshape((trainval_images.shape[0], 28*28))
trainval_images = trainval_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28*28))
test_images = test_images.astype('float32') / 255

# Categorically encode labels
trainval_labels = to_categorical(trainval_labels, 10)
test_labels = to_categorical(test_labels, 10)

# split training data and validation data
#train_images = trainval_images[0:50000]
#val_images = trainval_images[50000:]
#train_labels = trainval_labels[0:50000]
#val_labels = trainval_labels[50000:]

# use a small set to speed up learning
train_images = trainval_images[0:1000]
val_images = trainval_images[1000:1100]
train_labels = trainval_labels[0:1000]
val_labels = trainval_labels[1000:1100]


# build model with
# 2 dense hidden layers: variable number of nodes in the first dense layer, 8 nodes in the second dense layer
# 1 densen output layer with 10 nodes for the 10 classes
# learning_rate between 0.00001 and 0.001 with logarithmic sampling
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Dense(units=hp.Int("units",min_value=8,max_value=32,step=8), activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("learning_rate",  min_value=0.0001, max_value=0.01, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])
    return model

# initialize the tuner
tuner = keras_tuner.GridSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=25,
    executions_per_trial=1,
    overwrite=True,
    directory="C:/Users/Vera.Hollink/OneDrive - Hogeschool Inholland/Documents/My Documents/Deep Learning minor/Scripts/tuner output",
    project_name="tuner_example",
)
print(tuner.search_space_summary())

# run the tuner
tuner.search(train_images, train_labels, epochs=2, validation_data=(val_images, val_labels))

# output best hyperparameter values and build model using these values
best_hp = tuner.get_best_hyperparameters()[0]
print("Best hyperparamters: ", best_hp.values)
tuned_model = tuner.hypermodel.build(best_hp)

