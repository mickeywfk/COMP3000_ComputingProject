import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder

# Load the training dataset
training_dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/portScanning.csv"
training_dataset = pd.read_csv(training_dataset_path)

# Convert IP addresses to numerical labels using OrdinalEncoder
ip_encoder = OrdinalEncoder()
training_dataset[['Source IP', 'Destination IP']] = ip_encoder.fit_transform(training_dataset[['Source IP', 'Destination IP']])

# One-hot encode the 'TCP Flag' and 'Protocol' columns
flags_encoded = pd.get_dummies(training_dataset['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(training_dataset['Protocol'], prefix='protocol')
training_dataset = pd.concat([training_dataset, flags_encoded, protocol_encoded], axis=1)
training_dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Prepare the training dataset for training the model
X_train = training_dataset.drop(["Label"], axis=1)  # Use all columns except 'Label' as features
y_train = training_dataset["Label"]  # Predict the 'Label' column

# Create and train the random forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Load the dataset for prediction
prediction_dataset_path = "C:/Users/admin/Desktop/Project_VSC/Data/server_01.csv"
prediction_dataset = pd.read_csv(prediction_dataset_path)

# Convert IP addresses to numerical labels using the same OrdinalEncoder
prediction_dataset[['Source IP', 'Destination IP']] = ip_encoder.transform(prediction_dataset[['Source IP', 'Destination IP']])

# One-hot encode the 'TCP Flag' and 'Protocol' columns for the prediction dataset
flags_encoded = pd.get_dummies(prediction_dataset['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(prediction_dataset['Protocol'], prefix='protocol')
prediction_dataset = pd.concat([prediction_dataset, flags_encoded, protocol_encoded], axis=1)
prediction_dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Prepare the prediction dataset for making predictions
X_pred = prediction_dataset.drop(["Label"], axis=1)  # Use all columns except 'Label' as features
y_true = prediction_dataset["Label"]  # True labels for the prediction dataset

# Make predictions on the prediction dataset
y_pred = classifier.predict(X_pred)

# Calculate accuracy score
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy Score:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Generate classification report
report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)