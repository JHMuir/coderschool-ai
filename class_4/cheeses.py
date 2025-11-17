import keras 
import numpy as np
import pandas as pd
import sklearn

# ============= Step 1: Data Preprocessing =============
data = pd.read_csv("cheeses.csv")
print(f"Loaded {len(data)} cheeses!")
# Remove cheeses without a type
data = data.dropna(subset=['color'])

features = ["cheese","type","texture","rind","milk","flavor","aroma"]

# for col in features:
#     data[col] = data[col].fillna('unknown')

print(data[features])
data['text'] = data[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Turn text into numbers (counts how many times each word appears)
vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_features=200)
x = vectorizer.fit_transform(data['text']).toarray()
num_features = x.shape[1]

# Turn milk types into numbers (0, 1, 2, 3...)
label_encoder = sklearn.preprocessing.LabelEncoder()
y = label_encoder.fit_transform(data['color'])
num_classes = len(label_encoder.classes_)

# Split into training and testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# ============= Step 2: Training =============
model = keras.models.Sequential()

# Overfitting
model.add(keras.layers.Dense(64, activation='relu', input_shape=(num_features,)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(32, activation='relu')) 
model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# model.add(keras.layers.Dense(64, activation='relu', input_shape=(num_features,)))
# model.add(keras.layers.Dropout(0.3)) 
# model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if no improvement for 5 epochs
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stop]
)

# ============= Step 3: Inferencing =============
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

print("\nFirst 10 predictions:")
for i in range(10):
    actual = label_encoder.inverse_transform([y_test[i]])[0]
    predicted = label_encoder.inverse_transform([predicted_classes[i]])[0]
    cheese = data['cheese'][i]
    correct = "Y: " if actual == predicted else "N: "
    print(f"{correct} Cheese: {cheese} | Actual: {actual} | Predicted: {predicted}")

# How many did we get right?
accuracy = np.mean(predicted_classes == y_test) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
