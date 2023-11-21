import pandas as pd

data = pd.read_csv('new_discretized_data_kerala.csv', parse_dates=['time'])

first_range = data[(data['time'] >= '2020-01-01') & (data['time'] <= '2021-04-21')]
second_range = data[data['time'] > '2021-04-21']

first_range.to_csv('training_kerala.csv', index=False)
second_range.to_csv('testing_kerala.csv', index=False)
