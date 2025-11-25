import tensorflow as tf
import keras
import numpy as np

# ============= Step 1: Data =============
# Keeping the 1000 most common words in this dataset
# The words are stored as number keys
# y is 1 for positive, 0 for negative
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
print(x_train[0])


# Pad sequences to same length (transformers need fixed input size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# ============= Step 2: Training =============
def create_transformer_model():
    # Input Layer (200 words at a time)
    inputs = keras.Input(shape=(200,))
    
    # Embedding: Turn word IDs into dense vectors
    # 32-dimension point in space - similar words are closer in its space
    x = keras.layers.Embedding(10000, 32)(inputs)
    
    # Multi-Head Attention
    # This lets the model figure out which words are important, and which word should pay attention to each other
    # "The movie wasn't very good "
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=2,      # How many "attention perspectives" e.g. good vs bad, others 
        key_dim=32        # Size of attention vectors
    )(x, x)  # Query and Key are the same (self-attention)
    
    # Normalize
    # Taking the input and adding it the attention layer (residual connection)
    x = keras.layers.Add()([x, attention_output])
    x = keras.layers.LayerNormalization()(x)
    
    # Feed-forward network
    ff = keras.layers.Dense(32, activation='relu')(x)
    ff = keras.layers.Dense(32)(ff)
    
    # Normalize again
    x = keras.layers.Add()([x, ff])
    x = keras.layers.LayerNormalization()(x)
    
    # Pool all the words into one representation
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Output: Positive or Negative?
    # Single probability, the closer to 1, the more confident it is about it's positivity
    outputs = keras.layers.Dense(1, activation='sigmoid')(x) 
    
    return keras.Model(inputs=inputs, outputs=outputs)

model = create_transformer_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Positive vs Negative
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ============= Step 3: Inferencing =============
# Get a review from the test set

for i in range(10):
    test_review = x_test[i:i+1]
    prediction = model.predict(test_review)[0][0]
    print(f"Sentiment: {'Positive' if prediction > 0.5 else 'Negative'} Confidence: {prediction:.2%} ")

# Test accuracy
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f"Test Accuracy: {accuracy:.2%}")