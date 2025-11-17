# Turning off GPU acceleration here - not needed if your device has not been configured with CUDA, and can be ignored. 
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
import tensorflow
import matplotlib
import numpy

# TensorFlow: low-level machine learning library built by Google
# Keras: high-level machine learning library that is built on top of TensorFlow. To start, use Keras
# Matplotlib: "Plotting" images of digits
# NumPy: Fancy library that lets us handle arrays and math

# Neural network: Like a brain made of math that learns patterns
# Training: Teaching the computer by showing it thousands of examples
# Layers: Different stages of processing, like layers in a cake
# Neurons: Individual "decision makers" that vote on what they see

numpy.set_printoptions(linewidth=200, threshold=1000) # Makes our arrays print nicely later on 

# print(tf.__version__)

# Handwritten digits dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Current features range from 0 - 255
print(x_train[0])
print(y_train[0])
# Normalizing features to be between 0 - 1. Helps the model understand better by simplifying numbers 
x_train = keras.utils.normalize(x_train, axis=1);
x_test = keras.utils.normalize(x_test, axis=1);

# Doing a simple plot to show the letter (not required)
matplotlib.pyplot.imshow(x_train[0], cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.show()

# A Sequential model processes information and data in a forward line, in direct order. Like an assembly line. 
model = keras.models.Sequential()

# Input Layer (takes numerical data). Flattens our 2D array into a long 1D array
model.add(keras.layers.Flatten())

# Hidden Layers 
# Dense Layers: "Fully Connected" layer where every neuron connects to every neuron in the next layer
# Activation: simple rules that says "if negative, keep it 0; if positive keep it"
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))

# Output Layers
# Outputs an array with probabilites of what number it could be 
model.add(keras.layers.Dense(10, activation='softmax')) # 10 because the numbers range from 0 - 9

model.compile(optimizer='adam', # Good default to start with 
              loss='sparse_categorical_crossentropy', # How we are calculating error
              metrics=['accuracy']) # What to track 

model.fit(x_train, y_train, epochs=3) # actually training the model
model.save("our_model.keras")

# Probability distributions (Which number is most likely to be predicted?)
new_model = keras.models.load_model("our_model.keras")
predictions = new_model.predict(x_test)
print(predictions)


print(numpy.argmax(predictions[0]))
matplotlib.pyplot.imshow(x_test[0], cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.show()

