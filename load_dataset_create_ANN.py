
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU, PReLU, ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


print(tf.__version__)

dataset = pd.read_csv("Churn_Modelling.csv")
# print(dataset.head)

# ##Devide the dataset into dependent and independent features
x = dataset.iloc[:, 3:13]  # All rows and 3rd column to 12th column
y = dataset.iloc[:, 13]  # All rows and 13th column

# Feature engineering
geography = pd.get_dummies(x['Geography'], drop_first=True)
gender = pd.get_dummies(x['Gender'], drop_first=True)

# concatinate these values with dataframe
x = x.drop(['Geography', 'Gender'], axis=1)
x = pd.concat([x, geography, gender], axis=1)


# Spilliting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 : Create ANN
# Initialize ANN
classifier = Sequential()

# Adding input layer
# We have 11 inout nodes. check with X_train.shape.
# So we need 11 input nodes

classifier.add(Dense(units=11, activation='relu'))

# Adding 1st hidden layer
classifier.add(Dense(units=7, activation='relu'))  # lets say 7 neurons

# Addding 2nd hidden layer
classifier.add(Dense(units=6, activation='relu'))  # lets say 6 neurons

# Addding Dropout layer
# tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
# rate: Float between 0 and 1. Fraction of the input units to drop.
# seed: A Python integer to use as random seed.
classifier.add(Dropout(.2, input_shape=(2,)))

# Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid'))  # binery classifier


# classifier.compile(optimizer='adam', loss='binary_cross_entropy', metrics=['accuracy'])

# we have not intreduced learning rate. By default adam will use learning rate of 0.1
# if we want to use a different learning rate then

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
classifier.compile(optimizer=opt, loss='binary_crossentropy',
                   metrics=['accuracy'])


# Train without Early Stopping condition
# model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=10)


# Introducing the Early Stopping.
# If you want to stop the training because accuracy is not chaining.
# Stop training when a monitored metric has stopped improving.

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)
# Here we are focusing on val_loss. so if val_loss is not changing, then training should stop.
model_history = classifier.fit(X_train, y_train, validation_split=0.33,
                               batch_size=10, epochs=1000, callbacks=early_stopping)

model_history.history.keys()

# Summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 : Prediction and Evaluation
# Prediction the Test Set
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)  # if value is >0.5 predict as 1, else predict as 0

# Make the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion metercs : \n', cm)
# Calculate Accuracy
score = accuracy_score(y_pred, y_test)

print('Score : \n', score)


# Get the weights
print("Weights : \n", classifier.get_weight_paths())
