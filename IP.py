import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset with IP addresses
data = pd.read_csv('C:/Users/admin/Desktop/Project_VSC/Data/testingData_4.csv')

# Extract the IP addresses column
ip_addresses = data['Source IP', 'Destination IP']

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Reshape the IP addresses to match the expected input of the encoder
ip_addresses_reshaped = ip_addresses.values.reshape(-1, 1)

# Perform one-hot encoding on the IP addresses
encoded_ip_addresses = encoder.fit_transform(ip_addresses_reshaped)

# Create column names for the encoded features
column_names = encoder.get_feature_names(['Source IP', 'Destination IP'])

# Create a DataFrame with the encoded features
encoded_data = pd.DataFrame(encoded_ip_addresses, columns=column_names)

# Concatenate the original data with the encoded features
data_encoded = pd.concat([data, encoded_data], axis=1)

# Drop the original IP addresses column
data_encoded = data_encoded.drop('Source IP', 'Destination IP', axis=1)

# Now, your data contains the one-hot encoded IP addresses as separate columns
print(data_encoded.head())