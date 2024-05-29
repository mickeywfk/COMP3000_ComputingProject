import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import graphviz
import joblib

# Load the trained model
model_file_path = 'C:/Users/admin/Desktop/Project_VSC/Data/model_v3.pkl'
model = joblib.load(model_file_path)
#model = joblib.load('C:/Users/admin/Desktop/Project_VSC/Data/model.pkl')

# Load the new data for prediction
new_data = pd.read_csv('C:/Users/admin/Desktop/Project_VSC/Data/testingData_raw.csv')

# Preprocess the new data (assuming it has the same features as the training data)
X_new = new_data.drop(['Label', 'Packets'], axis=1)
X_new_encoded = pd.get_dummies(X_new, columns=['Source IP', 'Destination IP', 'Protocol'])

# Make predictions on the new data
y_pred = model.predict(X_new_encoded)

# Add the predictions to the new data DataFrame
new_data['Prediction'] = y_pred

# Save the predictions as a CSV file
prediction_file_path = 'C:/Users/admin/Desktop/Project_VSC/Data/predictions.csv'
new_data.to_csv(prediction_file_path, index=False)