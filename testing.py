import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare the training data
train_features_1 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # Example training data for Feature 1
train_features_2 = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])  # Example training data for Feature 2
train_targets = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])  # Example one-hot encoded training targets

# Reshape the input data
train_features_1 = train_features_1.reshape((train_features_1.shape[0], train_features_1.shape[1], 1))
train_features_2 = train_features_2.reshape((train_features_2.shape[0], train_features_2.shape[1], 1))

print(train_features_1)
print(train_features_2)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(2, 1)))  # Assumes each feature has 2 categories
model.add(Dense(3, activation='softmax'))  # 3 output classes for the target variable

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([train_features_1, train_features_2], train_targets, epochs=10, batch_size=1, verbose=1)

# Prepare the testing data
test_features_1 = np.array([[1, 0], [0, 1]])  # Example testing data for Feature 1
test_features_2 = np.array([[0, 1], [1, 0]])  # Example testing data for Feature 2

test_features_1 = test_features_1.reshape((test_features_1.shape[0], test_features_1.shape[1], 1))
test_features_2 = test_features_2.reshape((test_features_2.shape[0], test_features_2.shape[1], 1))

# Make predictions on the testing data
predictions = model.predict([test_features_1, test_features_2])
print(predictions)

# Decode the predictions
decoded_predictions = np.argmax(predictions, axis=1)
print("Predictions:", decoded_predictions)