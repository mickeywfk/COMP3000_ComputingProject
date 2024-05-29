import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

#Data Collection and Preparation
data = pd.read_csv('C:/Users/admin/Desktop/Project_VSC/Data/testingData_4.csv')  # Load the traffic data from a CSV file or any other data source

#Split the data into features and labels
X = data.drop(['Label', 'Packets'], axis=1)  # Features
y = data['Label']  # Labels

# Convert categorical columns to numerical representation (one-hot encoding)
X_encoded = pd.get_dummies(X, columns=['Source IP', 'Destination IP', 'Protocol'])  

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#Model Selection and Training
model = RandomForestClassifier()  # Create a random forest classifier
model.fit(X_train, y_train)  # Train the model on the training data

#Model Evaluation
y_pred = model.predict(X_test)  # Perform predictions on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
print('Accuracy:', accuracy)

# Additional outputs
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print('Confusion Matrix:')
print(confusion_mat)

print('Classification Report:')
print(classification_rep)

plt.figure()
plt.imshow(confusion_mat, interpolation="nearest")
plt.show()

# Extract a single decision tree from the random forest model
estimator = model.estimators_[0]

# Export the decision tree as a DOT file
dot_data = tree.export_graphviz(estimator, out_file=None, feature_names=X_encoded.columns, class_names=model.classes_, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)

# Save the graphical representation as a PDF file
graph_file_path = 'C:/Users/admin/Desktop/Project_VSC/Data/decision_tree.pdf'
graph.format = 'pdf'
graph.render(graph_file_path)