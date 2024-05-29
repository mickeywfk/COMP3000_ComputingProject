import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Prepare the training data
Time_data = ['5/5/2024 22:17:00', '5/5/2024 22:18:30', '5/5/2024 22:19:15', '5/5/2024 22:20:45']
train_features_1 = np.array([1, 2, 1, 2])  # Example training data for Feature 1
train_features_2 = np.array([0, 1, 1, 0])  # Example training data for Feature 2
train_targets = np.array([1, 2, 2, 1])  # Example training targets

# Prepare the training data with time steps and encoded time data
train_data = np.column_stack((train_features_1, train_features_2))

print(train_data)

# Encode the Time_data
encoded_time_data = np.array([[int(time.split()[1].split(':')[0]), int(time.split()[1].split(':')[1]), int(time.split()[1].split(':')[2])] for time in Time_data])

# Prepare the training data with time steps and encoded time data
train_data_with_time = np.column_stack((train_data, encoded_time_data))

print(train_data_with_time)

# Reshape the input data for LSTM
train_data_with_time = train_data_with_time.reshape((train_data_with_time.shape[0], train_data_with_time.shape[1], 1))

print(train_data_with_time)
print("....................")
print(train_data_with_time.shape[1])
print("....................")
print(train_data_with_time.shape[2])

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(train_data_with_time.shape[1], train_data_with_time.shape[2])))
model.add(Dense(3, activation='softmax'))  # 3 output classes for the target variable

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_data_with_time, train_targets, epochs=10, batch_size=1, verbose=1)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(train_data_with_time, train_targets, verbose=0)
print(f'Training Loss: {loss:.4f}')
print(f'Training Accuracy: {accuracy:.4f}')

# Prepare the testing data
test_time_data = ['5/6/2024 18:10:10', '5/6/2024 20:30:10', '5/6/2024 20:30:11', '5/6/2024 20:31:00']
test_features_1 = np.array([1, 2, 2, 1])  # Example testing data for Feature 1
test_features_2 = np.array([0, 1, 0, 0])  # Example testing data for Feature 2

test_data = np.column_stack((test_features_1, test_features_2))

encoded_test_time_data = np.array([[int(time.split()[1].split(':')[0]), int(time.split()[1].split(':')[1]), int(time.split()[1].split(':')[2])] for time in test_time_data])

test_data_with_time = np.column_stack((test_data, encoded_test_time_data))

# Make predictions on the testing data
predictions = model.predict(test_data_with_time)
decoded_predictions = np.argmax(predictions, axis=1)

# Plot the predicted labels
plt.scatter(range(len(decoded_predictions)), decoded_predictions, c='r', label='Predicted Labels')
plt.title('Predicted Labels')
plt.xlabel('Samples')
plt.ylabel('Labels')
plt.legend()
plt.show()

# Print the predictions
print("Predictions:", decoded_predictions)