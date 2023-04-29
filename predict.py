import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def load_image(img_name):
    # load input image
    input_image = Image.open(img_name)

    # resize the image to 28x28 pixels
    resized_image = input_image.resize((28, 28))

    # convert the image to grayscale
    grayscale_image = resized_image.convert('L')

    # normalize the pixel values to [0, 1]
    normalized_image = np.array(grayscale_image) / 255.0

    # reshape the image to have a single channel
    reshaped_image = normalized_image.reshape((1, 28, 28, 1))

    return reshaped_image

image = load_image('number.jpg')

# load the saved model
model = load_model('my_model.h5')

# use the model for prediction
predictions = model.predict(image)

result = np.argmax(predictions)

print(result)
