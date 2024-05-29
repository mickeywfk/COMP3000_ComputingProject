import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
dataset = pd.read_csv('C:/Users/admin/Desktop/Project_VSC/Data/testingData.csv')

# Replace the IP encoding code with custom encoding
local_device_ips = ['192.168.10.11']
trust_ips = ['192.168.1.1', '192.168.1.3', '192.168.1.12', '192.168.10.2']
encoder = OneHotEncoder()
label_encoder = LabelEncoder()

def encode_ip(ip):
    if ip in local_device_ips:
        return 'Local device'
    elif ip in trust_ips:
        return 'Trust IP'
    else:
        return 'Other IP'

# Apply the custom encoding to 'Source IP' and 'Destination IP' columns
dataset['Source IP'] = dataset['Source IP'].apply(encode_ip)
dataset['Destination IP'] = dataset['Destination IP'].apply(encode_ip)
dataset['Source Port'] = dataset['Source Port'].fillna(0)
dataset['Destination Port'] = dataset['Destination Port'].fillna(0)

# Prepare the training data
train_features_1 = np.array(dataset['Source IP'])
train_features_2 = np.array(dataset['Destination IP'])
train_features_3 = np.array(dataset['Source Port'])
train_features_4 = np.array(dataset['Destination Port'])
train_features_5 = np.array(dataset['TCP Flag'])
train_features_6 = np.array(dataset['Protocol'])
train_features_7 = np.array(dataset['Packets'])
train_targets = np.array(dataset['Label'])

# Encode categorical features using OneHotEncoder
train_features_1_encoded = encoder.fit_transform(train_features_1.reshape(-1, 1)).toarray()
train_features_2_encoded = encoder.fit_transform(train_features_2.reshape(-1, 1)).toarray()
train_features_5_encoded = encoder.fit_transform(train_features_5.reshape(-1, 1)).toarray()
train_features_6_encoded = encoder.fit_transform(train_features_6.reshape(-1, 1)).toarray()

# Convert labels to numerical values using LabelEncoder
train_targets_encoded = label_encoder.fit_transform(train_targets)

# Combine features into a single array
train_data = np.column_stack((train_features_1_encoded, train_features_2_encoded, train_features_3, train_features_4, train_features_5_encoded, train_features_6_encoded, train_features_7))
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))

model = Sequential()
model.add(Input(shape=(train_data.shape[1], train_data.shape[2])))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_targets_encoded, epochs=10, batch_size=1, verbose=1)

model_file_path = 'C:/Users/admin/Desktop/Project_VSC/Data/RNN_model.h5'
model.save(model_file_path)

# Plot the training loss
#plt.plot(history.history['loss'])
#plt.title('Training Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(train_data, train_targets_encoded, verbose=0)
print(f'Training Loss: {loss:.4f}')
print(f'Training Accuracy: {accuracy:.4f}')

model_file_path = 'C:/Users/admin/Desktop/Project_VSC/Data/RNN_model.h5'
model = load_model(model_file_path)

# Example prediction code
prediction_dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/trainingData.csv"
prediction_dataset = pd.read_csv(prediction_dataset_path)

# Apply the same preprocessing steps to the prediction dataset
prediction_dataset['Source IP'] = prediction_dataset['Source IP'].apply(encode_ip)
prediction_dataset['Destination IP'] = prediction_dataset['Destination IP'].apply(encode_ip)
prediction_dataset['Source Port'] = prediction_dataset['Source Port'].fillna(0)
prediction_dataset['Destination Port'] = prediction_dataset['Destination Port'].fillna(0)

prediction_features_1 = np.array(prediction_dataset['Source IP'])
prediction_features_2 = np.array(prediction_dataset['Destination IP'])
prediction_features_3 = np.array(prediction_dataset['Source Port'])
prediction_features_4 = np.array(prediction_dataset['Destination Port'])
prediction_features_5 = np.array(prediction_dataset['TCP Flag'])
prediction_features_6 = np.array(prediction_dataset['Protocol'])
prediction_features_7 = np.array(prediction_dataset['Packets'])

# Encode categorical features using OneHotEncoder
prediction_features_1_encoded = encoder.fit_transform(prediction_features_1.reshape(-1, 1)).toarray()
prediction_features_2_encoded = encoder.fit_transform(prediction_features_2.reshape(-1, 1)).toarray()
prediction_features_5_encoded = encoder.fit_transform(prediction_features_5.reshape(-1, 1)).toarray()
prediction_features_6_encoded = encoder.fit_transform(prediction_features_6.reshape(-1, 1)).toarray()

# Combine features into a single array
prediction_data = np.column_stack((prediction_features_1_encoded, prediction_features_2_encoded, prediction_features_3, prediction_features_4, prediction_features_5_encoded, prediction_features_6_encoded, prediction_features_7))
prediction_data = prediction_data.reshape((prediction_data.shape[0], prediction_data.shape[1], 1))

# Predict using the trained model
predictions = model.predict(prediction_data)

# Convert predictions to labels using inverse_transform
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Print the predicted labels
#print(predicted_labels)

# Convert predictions to labels using inverse_transform
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Create a DataFrame with the predicted labels
prediction_results = pd.DataFrame({'Predicted Label': predicted_labels})

prediction_targets_encoded = label_encoder.fit_transform(np.array(prediction_dataset['Label']))

# Save the DataFrame as a CSV file
prediction_results.to_csv('C:/Users/admin/Desktop/Project_VSC/Data/RNN_prediction_results.csv', index=False)

loss, accuracy = model.evaluate(prediction_data, prediction_targets_encoded, verbose=0)
print(f'Training Loss: {loss:.4f}')
print(f'Training Accuracy: {accuracy:.4f}')