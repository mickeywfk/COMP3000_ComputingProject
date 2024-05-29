import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare the training data
train_features_1 = np.array([1, 2, 1, 2])  # Example training data for Feature 1
train_features_2 = np.array([0, 1, 1, 0])  # Example training data for Feature 2
train_targets = np.array([1, 2, 2, 1])  # Example training targets

# Reshape the input data for LSTM
train_features_1 = train_features_1.reshape((train_features_1.shape[0], 1, 1))
train_features_2 = train_features_2.reshape((train_features_2.shape[0], 1, 1))

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(1, 1)))  # Assumes each feature has 1 timestep and 1 feature
model.add(Dense(3, activation='softmax'))  # 2 output classes for the target variable

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([train_features_1, train_features_2], train_targets, epochs=10, batch_size=1, verbose=1)

# Prepare the testing data
test_features_1 = np.array([1, 2])  # Example testing data for Feature 1
test_features_2 = np.array([0, 1])  # Example testing data for Feature 2

test_features_1 = test_features_1.reshape((test_features_1.shape[0], 1, 1))
test_features_1 = test_features_1.reshape((test_features_1.shape[0], 1, 1))

# Make predictions on the testing data
predictions = model.predict([test_features_1, test_features_2])
print(predictions)

# Decode the predictions
decoded_predictions = np.argmax(predictions, axis=1)
print("Predictions:", decoded_predictions)