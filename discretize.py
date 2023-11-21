import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

# Load the CSV file into a DataFrame
data = pd.read_csv('dataset-kerala.csv', parse_dates=['time'])

print(data.columns.tolist())

# Assuming 'data' contains the variables you want to discretize
columns_to_discretize = [col for col in data.columns if col not in ['time', 'latitude', 'longitude']]
data_to_discretize = data[columns_to_discretize]

# Initialize the KBinsDiscretizer to create bins
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')

# Fit and transform the data to discretize
discretized_data = discretizer.fit_transform(data_to_discretize)

# You can also use KMeans directly if you want to have more control over the process
# For instance, specifying the number of clusters (n_clusters)
# kmeans = KMeans(n_clusters=5)
# clusters = kmeans.fit_predict(data)
# Then, use the cluster labels as discretized values

# Convert the discretized array back to a DataFrame
discretized_df = pd.DataFrame(discretized_data, columns=columns_to_discretize)

# Save the discretized data to a new CSV file
non_discretized_columns = ['time', 'latitude', 'longitude']
final_df = pd.concat([data[non_discretized_columns], discretized_df], axis=1)

# Save the discretized data to a new CSV file
final_df.to_csv('discretized_data_kerala.csv', index=False)















