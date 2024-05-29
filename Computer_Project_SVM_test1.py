import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import time
from sklearn.metrics import accuracy_score

# Load the dataset
dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/testingData.csv"
dataset = pd.read_csv(dataset_path)
start_time = time.time()
# Preprocess the "Time" column
dataset['Time'] = pd.to_datetime(dataset['Time'], dayfirst=True).astype('int64')
#dataset['Time'] = pd.to_datetime(dataset['Time'], dayfirst=True).dt.strftime("%d/%m/%Y %H:%M")
#print(dataset['Time'])

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

# Convert remaining non-numeric columns using label encoding
non_numeric_columns = dataset.select_dtypes(include=['object'])

label_encoder = LabelEncoder()
for column in non_numeric_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Select feature columns and target variable
features = dataset.drop("Label", axis=1)
target = dataset["Label"]
#print(features)

processed_dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/processed_data.csv"
dataset.to_csv(processed_dataset_path, index=False)
start_time = time.time()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create an SVM classifier
svm = SVC(kernel='rbf', C=1.0, random_state=42)

# Train the SVM classifier
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Reverse the predicted labels
y_pred_reverse = label_encoder.inverse_transform(y_pred)
y_test_reverse = label_encoder.inverse_transform(y_test)

# Evaluate the classifier
print(classification_report(y_test_reverse, y_pred_reverse, zero_division=1))

loading_time = time.time() - start_time
print("Loading time:", loading_time, "seconds")

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame(columns=['Instance', 'True Label', 'Predicted Label'])

# Populate the DataFrame with predictions for individual instances
for i in range(len(X_test)):
    instance = i + 1
    true_label = y_test_reverse[i]
    predicted_label = y_pred_reverse[i]
    predictions_df.loc[i] = [instance, true_label, predicted_label]

# Save the predictions DataFrame to a CSV file
predictions_csv_path = "C:/Users/admin/Desktop/Project_VSC/Data/SVM_predictions.csv"
predictions_df.to_csv(predictions_csv_path, index=False)


# Load the new dataset for prediction
new_dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/trainingData.csv"
new_dataset = pd.read_csv(new_dataset_path)

start_time = time.time()

result = new_dataset["Label"]

# Preprocess the new dataset in the same way as the original dataset
new_dataset['Time'] = pd.to_datetime(new_dataset['Time'], dayfirst=True).astype('int64')
# Apply the custom encoding to 'Source IP' and 'Destination IP' columns
new_dataset['Source IP'] = new_dataset['Source IP'].apply(encode_ip)
new_dataset['Destination IP'] = new_dataset['Destination IP'].apply(encode_ip)
new_dataset['Source Port'] = new_dataset['Source Port'].fillna(0)
new_dataset['Destination Port'] = new_dataset['Destination Port'].fillna(0)

# Apply the custom encoding to 'TCP Flag' column
new_dataset['TCP Flag'] = new_dataset['TCP Flag'].apply(encode_flags)

# One-hot encode the modified 'Source IP' and 'Destination IP' columns
for i, flag in enumerate(flag_columns):
    new_dataset[flag] = new_dataset['TCP Flag'].apply(lambda x: x[i] if isinstance(x, list) else 0)

# Convert the lists of flags to separate columns
new_source_ip_encoded = pd.DataFrame(new_dataset['Source IP'].to_list(), columns=['Source Local Device', 'Source Trust IP', 'Source Other IP'])
new_destination_ip_encoded = pd.DataFrame(new_dataset['Destination IP'].to_list(), columns=['Destination Local Device', 'Destination Trust IP', 'Destination Other IP'])
new_dataset = pd.concat([new_dataset, new_source_ip_encoded, new_destination_ip_encoded], axis=1)
new_dataset.drop(['Source IP', 'Destination IP'], axis=1, inplace=True)

# One-hot encode the 'TCP Flag' and 'Protocol' columns
new_protocol_encoded = pd.get_dummies(new_dataset['Protocol'], prefix='protocol')
new_dataset = pd.concat([new_dataset, new_protocol_encoded], axis=1)
new_dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Convert remaining non-numeric columns using label encoding
for column in non_numeric_columns:
    new_dataset[column] = label_encoder.transform(new_dataset[column])

# Ensure the new dataset has the same columns as the original dataset used for training
new_dataset = new_dataset[features.columns]

# Use the trained SVM model to make predictions on the new dataset
new_predictions = svm.predict(new_dataset)

# Reverse the predicted labels
new_predictions_reverse = label_encoder.inverse_transform(new_predictions)

# Compare the predicted labels with the "Label" column in the new dataset
new_dataset["Predicted Label"] = new_predictions_reverse

# Compare the predicted labels with the "Label" column in the training dataset
label_comparison = new_dataset["Predicted Label"] == result

# Save the new dataset with predicted labels and label comparison results to a CSV file
new_dataset_with_predictions_csv_path = "C:/Users/admin/Desktop/Project_VSC/Data/new_dataset_with_predictions.csv"
new_dataset.to_csv(new_dataset_with_predictions_csv_path, index=False)

# Save the label comparison results to a separate CSV file
label_comparison_csv_path = "C:/Users/admin/Desktop/Project_VSC/Data/label_comparison.csv"
label_comparison.to_csv(label_comparison_csv_path, index=False)

accuracy = accuracy_score(y_test_reverse, y_pred_reverse)
accuracy_percentage = accuracy * 100
print("Prediction accuracy on testingData.csv:", accuracy_percentage, "%")

prediction_time = time.time() - start_time
print("Prediction time:", prediction_time, "seconds")