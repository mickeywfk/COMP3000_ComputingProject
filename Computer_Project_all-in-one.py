import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Load the dataset
dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/testingData.csv"
dataset = pd.read_csv(dataset_path)
start_time = time.time()

# Replace the IP encoding code with custom encoding
local_device_ips = ['192.168.10.11']
trust_ips = ['192.168.1.1', '192.168.1.3', '192.168.1.12', '192.168.10.2']
flag_columns = ['FIN', 'SYN', 'RST', 'PSH', 'ACK']

def encode_ip(ip):
    if ip in local_device_ips:
        return [1, 0, 0]  # Local device
    elif ip in trust_ips:
        return [0, 1, 0]  # Trust IP
    else:
        return [0, 0, 1]  # Other IP
    
def encode_flags(flags):
    if isinstance(flags, str):
        hex_number = flags[-2:]  # Extract the last two characters representing the hex number
        flags = int(hex_number, 16)  # Convert the hex number to an integer
        encoded_flags = [
            (flags & 0x01) >> 0,  # FIN
            (flags & 0x02) >> 1,  # SYN
            (flags & 0x04) >> 2,  # RST
            (flags & 0x08) >> 3,  # PSH
            (flags & 0x10) >> 4   # ACK
        ]
        return encoded_flags
    else:
        return flags

# Apply the custom encoding to 'Source IP' and 'Destination IP' columns
dataset['Source IP'] = dataset['Source IP'].apply(encode_ip)
dataset['Destination IP'] = dataset['Destination IP'].apply(encode_ip)
dataset['Source Port'] = dataset['Source Port'].fillna(0)
dataset['Destination Port'] = dataset['Destination Port'].fillna(0)

# Apply the custom encoding to 'TCP Flag' column
dataset['TCP Flag'] = dataset['TCP Flag'].apply(encode_flags)

# One-hot encode the modified 'Source IP' and 'Destination IP' columns
for i, flag in enumerate(flag_columns):
    dataset[flag] = dataset['TCP Flag'].apply(lambda x: x[i] if isinstance(x, list) else 0)
#dataset.drop(['Source IP', 'Destination IP'], axis=1, inplace=True)

# Convert the lists of flags to separate columns
source_ip_encoded = pd.DataFrame(dataset['Source IP'].to_list(), columns=['Source Local Device', 'Source Trust IP', 'Source Other IP'])
destination_ip_encoded = pd.DataFrame(dataset['Destination IP'].to_list(), columns=['Destination Local Device', 'Destination Trust IP', 'Destination Other IP'])
dataset = pd.concat([dataset, source_ip_encoded, destination_ip_encoded], axis=1)
dataset.drop(['Source IP', 'Destination IP'], axis=1, inplace=True)

# One-hot encode the 'TCP Flag' and 'Protocol' columns
# flags_encoded = pd.get_dummies(dataset['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(dataset['Protocol'], prefix='protocol')
dataset = pd.concat([dataset, protocol_encoded], axis=1)
dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)


# Prepare the dataset for training and testing the model
X = dataset.drop(["Label", "Time"], axis=1)
y = dataset["Label"]  # Predict the 'Label' column

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the random forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the training and testing subsets
train_predictions = classifier.predict(X_train)
test_predictions = classifier.predict(X_test)

# Calculate and print the accuracy score for training and testing
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Training Accuracy Score:", train_accuracy)
print("Testing Accuracy Score:", test_accuracy)
"""
# Calculate and print the confusion matrix for training and testing
train_confusion = confusion_matrix(y_train, train_predictions)
test_confusion = confusion_matrix(y_test, test_predictions)
print("Training Confusion Matrix:")
print(train_confusion)
print("Testing Confusion Matrix:")
print(test_confusion)
plt.figure()
plt.imshow(train_confusion, interpolation="nearest")
#plt.show()
plt.figure()
plt.imshow(test_confusion, interpolation="nearest")
#plt.show()

# Calculate and print the classification report for training and testing
train_classification_rep = classification_report(y_train, train_predictions)
test_classification_rep = classification_report(y_test, test_predictions)
print("Training Classification Report:")
print(train_classification_rep)
print("Testing Classification Report:")
print(test_classification_rep)
"""
loading_time = time.time() - start_time
print("Loading time:", loading_time, "seconds")

new_dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/RF_ProcessData.csv"
dataset.to_csv(new_dataset_path, index=False)

# Load the dataset for prediction
prediction_dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/trainingData.csv"
prediction_dataset = pd.read_csv(prediction_dataset_path)

start_time = time.time()

# Convert IP addresses to numerical labels
prediction_dataset['Source IP'] = prediction_dataset['Source IP'].apply(encode_ip)
prediction_dataset['Destination IP'] = prediction_dataset['Destination IP'].apply(encode_ip)
prediction_dataset['Source Port'] = prediction_dataset['Source Port'].fillna(0)
prediction_dataset['Destination Port'] = prediction_dataset['Destination Port'].fillna(0)

prediction_source_ip_encoded = pd.DataFrame(prediction_dataset['Source IP'].to_list(), columns=['Source Local Device', 'Source Trust IP', 'Source Other IP'])
prediction_destination_ip_encoded = pd.DataFrame(prediction_dataset['Destination IP'].to_list(), columns=['Destination Local Device', 'Destination Trust IP', 'Destination Other IP'])
prediction_dataset = pd.concat([prediction_dataset, prediction_source_ip_encoded, prediction_destination_ip_encoded], axis=1)
prediction_dataset.drop(['Source IP', 'Destination IP'], axis=1, inplace=True)

actual_labels = prediction_dataset['Label']

# One-hot encode the 'TCP Flag' and 'Protocol' columns for the prediction dataset
prediction_dataset['TCP Flag'] = prediction_dataset['TCP Flag'].apply(encode_flags)
for i, flag in enumerate(flag_columns):
    prediction_dataset[flag] = prediction_dataset['TCP Flag'].apply(lambda x: x[i] if isinstance(x, list) else 0)

protocol_encoded = pd.get_dummies(prediction_dataset['Protocol'], prefix='protocol')
prediction_dataset = pd.concat([prediction_dataset, protocol_encoded], axis=1)
prediction_dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Reorder columns to match training dataset
prediction_dataset = prediction_dataset[X_train.columns]

# Prepare the prediction dataset for making predictions
X_pred = prediction_dataset  # Use all columns as features

# Make predictions on the prediction dataset
predictions = classifier.predict(X_pred)

# Create a copy of the prediction dataset and assign the predictions
prediction_results = prediction_dataset.copy()
prediction_results['Prediction'] = predictions

# Add the original labels to the prediction results
prediction_results['Original Label'] = actual_labels

# Save the prediction results as a CSV file
output_path = "C:/Users/admin/Desktop/Project_VSC/Data/RF_prediction_results.csv"
prediction_results.to_csv(output_path, index=False)

# Calculate the accuracy of the model's predictions on the prediction dataset
accuracy = accuracy_score(actual_labels, predictions)
print("Accuracy on trainingData.csv:", accuracy)

prediction_time = time.time() - start_time
print("Prediction time:", prediction_time, "seconds")