import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import OneHotEncoder

# Prepare the training data
train_features_1 = np.array(["Left", "Right", "Left", "Right"])  # Example training data for Feature 1
train_features_2 = np.array(["Red", "Blue", "Blue", "Red"])  # Example training data for Feature 2
train_targets = np.array([1, 2, 2, 1])  # Example training targets

# Map target values to labels
train_targets_mapped = np.where(train_targets == 1, "Good", "Bad")

# Perform one-hot encoding
encoder = OneHotEncoder()
train_features_1_encoded = encoder.fit_transform(train_features_1.reshape(-1, 1)).toarray()
train_features_2_encoded = encoder.fit_transform(train_features_2.reshape(-1, 1)).toarray()

print(train_features_1_encoded)
print(train_features_2_encoded)

# Define the model architecture
model = Sequential()
model.add(Dense(128, input_dim=train_features_1_encoded.shape[1], activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 output classes for the target variable

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([train_features_1_encoded, train_features_2_encoded], train_targets, epochs=10, batch_size=1, verbose=1)


# Prepare the testing data
test_features_1 = np.array(["Right", "Left"])  # Example testing data for Feature 1
test_features_2 = np.array(["Red", "Blue"])  # Example training data for Feature 1

test_features_encode_1 = encoder.fit_transform(test_features_1.reshape(-1, 1)).toarray()
test_features_encode_2 = encoder.fit_transform(test_features_2.reshape(-1, 1)).toarray()

# Make predictions on the testing data
predictions = model.predict([test_features_encode_1, test_features_encode_2])
print(predictions)

# Decode the predictions
decoded_predictions = np.where(np.argmax(predictions, axis=1) == 0, "Good", "Bad")
print("Predictions:", decoded_predictions)
