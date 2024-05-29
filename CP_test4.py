import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

# Load the training dataset
training_dataset_path = "portScanning.csv"
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

# Load the dataset to be predicted
prediction_dataset_path = "path_to_prediction_dataset.csv"
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

# Make predictions on the prediction dataset
predictions = classifier.predict(X_pred)

# Add a new column for predictions
prediction_dataset['Prediction'] = predictions

# Save the dataset with predictions to a new CSV file
prediction_dataset.to_csv('dataset_with_predictions.csv', index=False)