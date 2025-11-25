# Turning off GPU acceleration here - not needed if your device has not been configured with CUDA, and can be ignored. 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
import tensorflow
import matplotlib
import numpy

# TensorFlow: low-level machine learning library built by Google. Highly optimized, with a lot of room for error. 
# Keras: high-level machine learning library that is built on top of TensorFlow. To start, use Keras
# Matplotlib: "Plotting" images of digits
# NumPy: Fancy library that lets us handle arrays and math

# Neural network: Like a brain made of math that learns patterns
# Training: Teaching the computer by showing it thousands of examples
# Layers: Different stages of processing, like layers in a cake
# Neurons: Individual "decision makers" that vote on what they see

numpy.set_printoptions(linewidth=200, threshold=1000) # Makes our arrays print nicely later on. Not really needed, but helps visualize the data when we print it! 

# print(tf.__version__)

# ===== Step 1: DataPreprocessing =====
# Handwritten digits dataset
mnist = keras.datasets.mnist # Grabbing a dataset directly from Keras
# We split our dataset into training and testing. We use the training dataset to, train, and the testing dataset later to validate the accuracy of our model firsthand
# x is our inputs, y is our output (the answer). 

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# Current individual features (pixels) range from 0 - 255
print(x_train[0]) # The input for the first instance of data. 28x28 grid of pixels. The bigger a number on the pixel, the "darker" that pixel is. 
print(y_train[0]) # The answer for the first instance of data. A single digit, 0-9.

# Normalizing features to be between 0 - 1. Helps the model understand better and train faster by simplifying numbers 
x_train = keras.utils.normalize(x_train, axis=1);
x_test = keras.utils.normalize(x_test, axis=1);

# Doing a simple plot to show the letter (not required)
matplotlib.pyplot.imshow(x_train[0], cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.show()

# ===== Step 2: Training ===== 
# A Sequential model processes information and data in a forward line, in direct order. Like an assembly line. 
# A NN contains 3 sections: The input layer, the output layer, and the hidden layers (everything inbetween, basically)
model = keras.models.Sequential()

# Input Layer (takes numerical data). Flattens our 2D array into a long 1D array
model.add(keras.layers.Flatten())

# Hidden Layers 
# Dense Layers: "Fully Connected" layer where every neuron connects to every neuron in the next layer
# ReLU Activation: simple rules that says "if negative, keep it 0; if positive keep it"
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))

# Output Layers
# Softmax Activation: Outputs an array with probabilites of what number it could be (NOT the number itself)
model.add(keras.layers.Dense(10, activation='softmax')) # 10 because the numbers range from 0 - 9

# After we architect our model, we then almost always run compile() and fit()
# compile() essentially tells the model HOW to train itself. Think of it like embedding settings
# fit() tells to the model TO train itself. This is the often lengthy part that can take a while depending on the task. 

model.compile(
    optimizer='adam', # Good default to start with 
    loss='sparse_categorical_crossentropy', # How we are calculating error
    metrics=['accuracy'] # What to prioritize during training
)

model.fit(
    x_train, # Passing our input and output data
    y_train, 
    epochs=3 # Epochs are the amount of times our model will comb over our entire dataset.
) # actually training the model
model.save("our_model.keras")

# Probability distributions (Which number is most likely to be predicted?)
new_model = keras.models.load_model("our_model.keras")
predictions = new_model.predict(x_test)
print(predictions) # An array of probabilities, with the index/position in the array being the number itself 


print(numpy.argmax(predictions[0])) # Find the maxinum/highest probability
matplotlib.pyplot.imshow(x_test[0], cmap=matplotlib.pyplot.cm.binary)
matplotlib.pyplot.show()

