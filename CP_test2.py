import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# Exclude unknown IP addresses from the prediction dataset
prediction_dataset = prediction_dataset[prediction_dataset['Source IP'].isin(ip_encoder.categories_[0])]
prediction_dataset = prediction_dataset[prediction_dataset['Destination IP'].isin(ip_encoder.categories_[1])]

# Convert IP addresses to numerical labels
prediction_dataset[['Source IP', 'Destination IP']] = ip_encoder.transform(prediction_dataset[['Source IP', 'Destination IP']])

# One-hot encode the 'TCP Flag' and 'Protocol' columns for the prediction dataset
flags_encoded = pd.get_dummies(prediction_dataset['TCP Flag'], prefix='flags')
protocol_encoded = pd.get_dummies(prediction_dataset['Protocol'], prefix='protocol')
prediction_dataset = pd.concat([prediction_dataset, flags_encoded, protocol_encoded], axis=1)
prediction_dataset.drop(['TCP Flag', 'Protocol'], axis=1, inplace=True)

# Ensure feature names match those used during training
missing_columns = set(X_train.columns) - set(prediction_dataset.columns)
for column in missing_columns:
    prediction_dataset[column] = 0

# Reorder columns to match training dataset
prediction_dataset = prediction_dataset[X_train.columns]

# Prepare the prediction dataset for making predictions
X_pred = prediction_dataset  # Use all columns as features

# Make predictions on the prediction dataset
predictions = classifier.predict(X_pred)

# Add the predictions as a new column in the prediction dataset
prediction_dataset['Prediction'] = predictions

# Save the prediction dataset with the added 'Prediction' column as a CSV file
output_path = "C:/Users/admin/Desktop/Project_VSC/Data/prediction_results.csv"
prediction_dataset.to_csv(output_path, index=False)

# Print a message to indicate the file has been saved
print(f"Prediction results saved to: {output_path}")