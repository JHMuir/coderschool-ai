import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
import sklearn # scikit-learn - machine learning library that provides tools to alter data
# Pandas
import pandas as pd # Pandas - library for handling spreadsheet-like data


# ============= Step 1: Data Preprocessing =============
data = pd.read_csv("Pokemon.csv")
print(f"Loaded {len(data)} Pokemon!")

# Viewing our data
print(data.head())
print(data.columns.tolist())

# Defining our training features
features = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
x = data[features].values
print(x)

# Defining our target features (what we want to predict)
y = data["Total"].values

# Splitting our data up
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# Normalizing our data between scales 
# Different stats might have different scales, so this normalizes them to same range
scaler = sklearn.preprocessing.StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ============= Step 2: Training =============
model = keras.models.Sequential()

# Input Layer - small, linear relationships
model.add(keras.layers.Dense(128, activation='relu', input_shape=(6,)))

# Hidden layers - these learn patterns
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))

# Instead of 10 outputs (digits 0-9), we have 1 output (the Total stat number)
model.add(keras.layers.Dense(1))

model.compile(
    optimizer='adam',                          
    loss='mean_squared_error', # Tells the model to severely punish big mistakes
    metrics=['mean_absolute_error']         
)

history = model.fit(
    x_train_scaled, y_train,
    epochs=100,                    # More epochs than MNIST since we have less data
    batch_size=32,
    validation_split=0.2,          # Use some training data to check progress
    verbose=1
)

# ============= Step 3: Inferencing =============
predictions = model.predict(x_test_scaled)

for i in range(10):
    actual = y_test[i]
    predicted = predictions[i][0] # This is 2D
    error = abs(actual - predicted)
    print(f"Pokemon {i+1}: Actual={actual:.0f}, Predicted={predicted:.1f}, Error={error:.1f}")
    
# ============= Plots =============

# Scattered points show where the model struggles
# Ideally, this would look like a diagonal line
plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Total Stats')
plt.ylabel('Predicted Total Stats')
plt.title('Actual vs Predicted Pokemon Stats')

# Showing how loss decreasing over time 
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Training Progress')
plt.legend()

plt.show()

custom_pokemon = np.array([[
    80,   # HP
    100,  # Attack  
    70,   # Defense
    85,   # Sp. Atk
    75,   # Sp. Def
    90    # Speed
]])

# Normalizing!
custom_pokemon_scaled = scaler.transform(custom_pokemon)

# Make prediction
predicted_total = model.predict(custom_pokemon_scaled)[0][0]
actual_sum = custom_pokemon[0].sum()
print(f"Custom Stat Pokemon- Predicted: {predicted_total}, Actual: {actual_sum}")
