import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
import emnist
# Preset Matplotlib figure sizes.
matplotlib.rcParams['figure.figsize'] = [9, 6]

import tensorflow as tf
print(tf.__version__)
tf.random.set_seed(22)

print("Load Train Data")
x_train, y_train = emnist.extract_training_samples('byclass')
print("Train Data Loaded")

for i in range(9):
    plt.subplot(3,3,1+i)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"True Label: {y_train[i]}")
    plt.subplots_adjust(hspace=.5)

print("Normalize Training Data")
x_train = tf.keras.utils.normalize(x_train, axis=1)
print("Training Data Normalized")

print("Training Data MLP with 2 hidden layer")
num_input = 28*28
num_hidden_1 = 500
num_hidden_2 = 500
num_output = 10

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(num_hidden_1, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(num_hidden_2, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(num_output, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
model.fit(x_train, y_train, epochs=20)
model.save('MLP_EMNIST.model')
print("Training model built successfully")

print("Test Dataset Loading")
x_test, y_test = emnist.extract_test_samples('byclass')
print("Test Dataset Loaded")

print("Normalizing Test Dataset")
x_test = tf.keras.utils.normalize(x_test, axis=1)

print("Testing Data")
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Value Loss: ",val_loss.round(4))
print("Accuracy: ",val_acc.round(4)*100)

'''plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()'''