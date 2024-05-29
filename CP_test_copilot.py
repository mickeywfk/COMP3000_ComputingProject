import pandas as pd
from scapy.layers.inet import TCP, IP
from scapy.all import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder

# Prepare your dataset in CSV format with features and labels
dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/testingData_5.csv"

# Load the dataset
dataset = pd.read_csv(dataset_path)

# Convert IP addresses to numerical labels using OrdinalEncoder
ip_encoder = OrdinalEncoder()
dataset[['Source IP', 'Destination IP']] = ip_encoder.fit_transform(dataset[['Source IP', 'Destination IP']])

# One-hot encode the 'flags' column
flags_encoded = pd.get_dummies(dataset['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(dataset['Protocol'], prefix='protocol')
dataset = pd.concat([dataset, flags_encoded, protocol_encoded], axis=1)
dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Split the dataset into features (X) and labels (y)
X = dataset.drop(["Info"], axis=1)
y = dataset["Info"]

# Create and train the random forest classifier
classifier = RandomForestClassifier()
classifier.fit(X, y)

# Predict labels for the entire dataset
predictions = classifier.predict(X)

# Convert predictions to boolean values (True or False)
is_port_scanning = predictions == 'Port Scan'

# Add a new column indicating port scanning or not
dataset['Port Scanning'] = is_port_scanning

# Convert boolean values to True or False
dataset['Port Scanning'] = dataset['Port Scanning'].astype(bool)

# Save the modified dataset to a new CSV file
dataset.to_csv('C:/Users/admin/Desktop/Project_VSC/Data/dataset_with_scanning.csv', index=False)