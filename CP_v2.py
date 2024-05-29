import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder

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

# Split the dataset into features and labels
X = dataset.drop(["Label"], axis=1)  # Use all columns except 'Label' as features
y = dataset["Label"]  # Labels

# Create and train the random forest classifier
classifier = RandomForestClassifier()
classifier.fit(X, y)

# Make predictions on the dataset
y_pred = classifier.predict(X)

# Calculate accuracy score
accuracy = accuracy_score(y, y_pred)
print("Accuracy Score:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate classification report
report = classification_report(y, y_pred)
print("Classification Report:")
print(report)