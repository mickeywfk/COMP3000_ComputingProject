import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("C:/Users/admin/Desktop/Project_VSC/Data/newTestingData.csv")

# Preprocessing the data
data['TCP Flag'].fillna('Unknown', inplace=True)  # Fill missing values with 'Unknown'
data['Protocol'].fillna('Unknown', inplace=True)
data.dropna(subset=['Destination Port'], inplace=True)  # Remove rows with missing 'Destination Port'

# Convert categorical features to numerical
data['TCP Flag'] = data['TCP Flag'].astype('category').cat.codes
print(data['TCP Flag'])
data['Protocol'] = data['Protocol'].astype('category').cat.codes
print(data['Protocol'])

'''
# Splitting into input features and target variable
X = data[['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'TCP Flag', 'Protocol', 'Bytes']]
y = data['Port Scanning']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predicting on the test set
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
'''