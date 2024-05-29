import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Prepare the training data
train_features_1 = np.array([1, 2, 1, 2])  # Example training data for Feature 1
train_features_2 = np.array([0, 1, 0, 1])  # Example training data for Feature 2
train_targets = np.array([0, 1, 1, 0])  # Example training targets

# Define the model architecture
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))  # Assumes each feature has 1 dimension
model.add(Dense(2, activation='softmax'))  # 2 output classes for the target variable

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([train_features_1, train_features_2], train_targets, epochs=10, batch_size=4, verbose=1)

# Prepare the testing data
test_features_1 = np.array([1, 2])  # Example testing data for Feature 1
test_features_2 = np.array([0, 1])  # Example testing data for Feature 2

# Make predictions on the testing data
predictions = model.predict([test_features_1, test_features_2])
print(predictions)

# Decode the predictions
decoded_predictions = np.argmax(predictions, axis=1)
print("Predictions:", decoded_predictions)