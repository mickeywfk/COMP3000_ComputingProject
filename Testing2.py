import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/portScanning_Time.csv"
data = pd.read_csv(dataset_path)

# Preprocess the "Time" column
data['Time'] = pd.to_datetime(data['Time']).astype('int64')

print(data['Time'])

# Preprocess the IP address columns using one-hot encoding
ip_columns = ["Source IP", "Destination IP"]
data_encoded = pd.get_dummies(data, columns=ip_columns)

# One-hot encode the 'TCP Flag' and 'Protocol' columns
flags_encoded = pd.get_dummies(data['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(data['Protocol'], prefix='protocol')
data = pd.concat([data, flags_encoded, protocol_encoded], axis=1)
data.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Convert remaining non-numeric columns using label encoding
non_numeric_columns = data.select_dtypes(include=['object'])
label_encoder = LabelEncoder()
for column in non_numeric_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Select feature columns and target variable
features = data.drop("Label", axis=1)
target = data["Label"]

# Impute missing values with column mean
features = features.fillna(features.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm = SVC(kernel='rbf', C=1.0, random_state=42)

# Train the SVM classifier
svm.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test_scaled)

# Evaluate the classifier
print(classification_report(y_test, y_pred))

'''
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
'''