import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

data = pd.read_csv('dataset-kerala.csv', parse_dates=['time'])


columns_to_discretize = [col for col in data.columns if col not in ['time', 'latitude', 'longitude']]


bins_per_column = {'msl': 3, 'tcc': 4, 'tp': 2, 'r': 3, 't': 2}


discretized_df = pd.DataFrame()


discretizer = KBinsDiscretizer(encode='ordinal', strategy='kmeans')


for column in columns_to_discretize:
    column_data = data[[column]]
    num_bins = bins_per_column.get(column, 3)  
    discretizer.n_bins = num_bins  
    discretized_data = discretizer.fit_transform(column_data)
    discretized_df[column] = discretized_data.reshape(-1)

non_discretized_columns = ['time', 'latitude', 'longitude']
final_df = pd.concat([data[non_discretized_columns], discretized_df], axis=1)

final_df.to_csv('new_discretized_data_kerala.csv', index=False)
