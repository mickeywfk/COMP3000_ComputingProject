import pandas as pd
from scapy.layers.inet import TCP, IP
from scapy.all import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Load the dataset
dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/portScanning.csv"
dataset = pd.read_csv(dataset_path)

# Convert IP addresses to numerical labels using OrdinalEncoder
ip_encoder = OrdinalEncoder()
dataset[['Source IP', 'Destination IP']] = ip_encoder.fit_transform(dataset[['Source IP', 'Destination IP']])

# One-hot encode the 'TCP Flag' and 'Protocol' columns
flags_encoded = pd.get_dummies(dataset['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(dataset['Protocol'], prefix='protocol')
dataset = pd.concat([dataset, flags_encoded, protocol_encoded], axis=1)
dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Prepare the dataset for prediction
X = dataset.drop(["Label"], axis=1)  # Use all columns except 'Label' as features
y = dataset["Label"]  # Predict the 'Label' column

# Create and train the random forest classifier
classifier = RandomForestClassifier()
classifier.fit(X, y)

model_file_path = 'C:/Users/admin/Desktop/Project_VSC/Data/portScanningModel_v1.pkl'
joblib.dump(classifier, model_file_path)

# Make predictions on the dataset
#predictions = classifier.predict(X)

# Add a new column for predictions
#dataset['Prediction'] = predictions

# Save the modified dataset to a new CSV file
#dataset.to_csv('C:/Users/admin/Desktop/Project_VSC/Data/dataset_with_predictions.csv', index=False)