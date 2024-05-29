import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the dataset
dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/portScanning_Time.csv"
dataset = pd.read_csv(dataset_path)

# Convert 'Time' column to UNIX timestamps
dataset['Time'] = pd.to_datetime(dataset['Time'], format='%d/%m/%Y %H:%M').values.astype(np.int64) // 10 ** 9
print(dataset['Time'])

#dataset['Time'] = pd.to_datetime(dataset['Time'], dayfirst=True).dt.strftime("%d/%m/%Y %H:%M")
#print(dataset['Time'])

# Convert IP addresses to numerical labels using OrdinalEncoder
ip_encoder = OrdinalEncoder()
dataset[['Source IP', 'Destination IP']] = ip_encoder.fit_transform(dataset[['Source IP', 'Destination IP']])

# One-hot encode the 'TCP Flag' and 'Protocol' columns
flags_encoded = pd.get_dummies(dataset['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(dataset['Protocol'], prefix='protocol')
dataset = pd.concat([dataset, flags_encoded, protocol_encoded], axis=1)
dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Prepare the dataset for training and testing the model
X = dataset.drop(["Time", "Label"], axis=1)  # Use all columns except 'Time' and 'Label' as features
y = dataset["Label"]  # Predict the 'Label' column

# Split the dataset into training and testing subsets
print(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to fit the RNN model
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# Create the RNN model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on the training and testing subsets
train_predictions = model.predict_classes(X_train)
test_predictions = model.predict_classes(X_test)

# Calculate and print the accuracy score for training and testing
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Training Accuracy Score:", train_accuracy)
print("Testing Accuracy Score:", test_accuracy)

# Calculate and print the confusion matrix for training and testing
train_confusion = confusion_matrix(y_train, train_predictions)
test_confusion = confusion_matrix(y_test, test_predictions)
print("Training Confusion Matrix:")
print(train_confusion)
print("Testing Confusion Matrix:")
print(test_confusion)