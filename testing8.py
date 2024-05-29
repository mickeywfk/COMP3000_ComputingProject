import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Load the dataset
dataset = pd.read_csv('C:/Users/admin/Desktop/Project_VSC/Data/portScanning_Time.csv')

# Replace the IP encoding code with custom encoding
local_device_ips = ['192.168.10.11']
trust_ips = ['192.168.1.1', '192.168.1.3', '192.168.1.12']

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

# One-hot encode the modified 'Source IP' and 'Destination IP' columns
encoded_ips = pd.get_dummies(dataset[['Source IP', 'Destination IP']])
dataset = pd.concat([dataset, encoded_ips], axis=1)
dataset.drop(['Source IP', 'Destination IP'], axis=1, inplace=True)

# Concatenate the 'TCP Flag' columns with the dataset
dataset = pd.concat([dataset, pd.get_dummies(dataset['TCP Flag'], prefix='Flag')], axis=1)
dataset.drop('TCP Flag', axis=1, inplace=True)

dataset['Source Port'] = dataset['Source Port'].fillna(0).astype(int)
dataset['Destination Port'] = dataset['Destination Port'].fillna(0).astype(int)

# Convert specific columns to NumPy arrays
columns_to_convert = ['Source Port', 'Destination Port', 'Packets']
dataset_np = dataset[columns_to_convert].to_numpy()

# Access the converted columns as NumPy arrays
source_port_np = dataset_np[:, 0]
destination_port_np = dataset_np[:, 1]
packet_np = dataset_np[:, 2]

# Separate feature columns and target column
X = np.concatenate((dataset.drop(columns_to_convert, axis=1).values, source_port_np[:, np.newaxis],
                    destination_port_np[:, np.newaxis], packet_np[:, np.newaxis]), axis=1)
y = dataset['Label'].values

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to match the input shape of the LSTM model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)





