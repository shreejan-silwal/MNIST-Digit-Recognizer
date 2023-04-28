import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import ssl

# disable ssl verification (not recommended if the code runs without this part)
ssl._create_default_https_context = ssl._create_unverified_context

# load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# reshape the data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# perform one-hot encoding for classification
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# define lists to store loss values
train_loss = []
val_loss = []

# Train the model and store loss values
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# plot the training and validation loss
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

# Make predictions
predictions = model.predict(X_test[:5])
print(np.argmax(predictions, axis=1))