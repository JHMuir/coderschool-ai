import tensorflow
import keras
import numpy
""" 
Transformers can create new content by learning patterns. 
The model learns Shakespeare's writing style and generates similar text. 
"""
# ============= Step 1: Data Preprocessing =============
path = keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the text
with open(path, 'r', encoding='utensorflow-8') as f:
    text = f.read()

print(f"Loaded {len(text):,} characters")
print(f"First 200 characters:\n{text[:200]}\n")

vocab = sorted(set(text)) # Every unique character 
vocab_size = len(vocab)
print(f"Unique characters: {vocab_size}")
print(f"Characters: {''.join(vocab)}\n")

# Create mapping from character to number and back
char_to_num = {char: num for num, char in enumerate(vocab)} # Letters to Numbers
num_to_char = {num: char for num, char in enumerate(vocab)} # Numbers to Letters

# Convert the entire text to numbers
text_as_int = numpy.array([char_to_num[c] for c in text])

# Creating Training Examples
# We'll teach the model: given 40 characters, predict the next character
seq_length = 40
examples_per_epoch = len(text) // (seq_length + 1)

# Create sequences: "Hello worl" -> predict "d"
# How do we provide examples for text generation? 
# Give it partial sentences and tech it to predict what comes next
def create_sequences(text_as_int, seq_length):
    sequences = []
    next_chars = []
    
    # Take every possible sequence of seq_length
    for i in range(0, len(text_as_int) - seq_length, 3):  # Step by 3 for speed
        sequences.append(text_as_int[i:i + seq_length])
        next_chars.append(text_as_int[i + seq_length])
    return numpy.array(sequences), numpy.array(next_chars)

x_train, y_train = create_sequences(text_as_int, seq_length)
print(f"Created {len(x_train):,} training examples")
print(f"input shape: {x_train.shape}, Output shape: {y_train.shape}\n")

# ============= Step 2: Building the Model =============
def create_text_transformer(vocab_size, seq_length):
    """
    This transformer learns to predict the next character by:
    1. Looking at patterns in the sequence
    2. Figuring out which previous characters are important
    3. Using that to guess what comes next
    The transformer looks at the 40-character sequence and learns:
    1. Which previous characters are important for predicting next ones
    2. Long-range patterns (like matching quotation marks)
    3. Language structure (common word endings, punctuation rules)
    """
    inputs = keras.input(shape=(seq_length,))
    
    # Embedding: Turn character IDs into rich vectors
    # Each character gets a 128-dimensional representation
    x = keras.layers.Embedding(vocab_size, 128)(inputs)
    
    # Position encoding: Help model know where each character is
    # Without this, "cat" and "tac" would look the same!
    # Transformers sees all characters vectors at once in no particular order. 
    # Ex) "cat" - without knowing the order, it could be "cat", "act", or "tac".
    positions = tensorflow.range(start=0, limit=seq_length, delta=1)
    position_embedding = keras.layers.Embedding(seq_length, 128)(positions)
    x = x + position_embedding
    
    # Multi-head attention lets model focus on different aspects
    # One head might focus on spaces, another on vowels
    # Ex: Knowing after 'q', 'u' is immediately after. Another head might focus on phrasing, the fact that sentences start with a capital, etc
    attention1 = keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=128
    )(x, x)
    x = keras.layers.Add()([x, attention1])
    x = keras.layers.LayerNormalization()(x)
    
    # Feed-forward: Process the attended information
    ff1 = keras.layers.Dense(256, activation='relu')(x)
    ff1 = keras.layers.Dense(128)(ff1)
    x = keras.layers.Add()([x, ff1])
    x = keras.layers.LayerNormalization()(x)
    
    # Learn even deeper patterns
    # Abstract patterns; rhymes, structure, etc 
    attention2 = keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=128
    )(x, x)
    x = keras.layers.Add()([x, attention2])
    x = keras.layers.LayerNormalization()(x)
    
    ff2 = keras.layers.Dense(256, activation='relu')(x)
    ff2 = keras.layers.Dense(128)(ff2)
    x = keras.layers.Add()([x, ff2])
    x = keras.layers.LayerNormalization()(x)
    
    # Take only the last character's representation
    # We only care about predicting what comes after the sequence
    x = x[:, -1, :]
    
    # Output: Probability for each possible next character
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

model = create_text_transformer(vocab_size, seq_length)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# ============= Step 3: Inferencing =============
def generate_text(model, phrase, length, temperature):
    
    seq_length = model.input_shape[1]
    input_phrase = [char_to_num[c] for c in phrase]
    input_phrase = tensorflow.expand_dims(input_phrase, 0)
    print(input_phrase)
    
    generated = []
    
    for i in range(length):
        if input_phrase.shape[1] > seq_length:
            input_phrase = input_phrase[:, -seq_length:]
        elif input_phrase.shape[1] < seq_length:
            pad_len = seq_length - input_phrase.shape[1]
            input_phrase = tensorflow.pad(input_phrase, [[0, 0], [pad_len, 0]])
            
        predictions = model(input_phrase)
        predictions = tensorflow.squeeze(predictions, 0)
        
        predictions = predictions / temperature
        predicted_id = tensorflow.random.categorical(tensorflow.math.log(tensorflow.expand_dims(predictions, 0)), num_samples=1)[0, 0].numpy()
        
        input_phrase = tensorflow.concat([input_phrase, [[predicted_id]]], axis=1)
        
        generated.append(num_to_char[predicted_id])
        
    return phrase + "".join(generated)

start_phrases = ["ROMEO: ", "To be or not to be", "What light through", "Where art thou"]

for phrase in start_phrases:
    
    for temp in [0.5, 1.0, 1.5]:
        generated = generate_text(model, phrase, length=150, temperature=temp)
        print(f"\nTemperature {temp}:")
        print(generated + "\n")
