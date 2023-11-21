import pandas as pd
from pgmpy.estimators import MmhcEstimator


print("begin")

data = pd.read_csv('training_kerala.csv')
print("loaded into pandas df")
data.drop(['latitude', 'longitude', 'time'], inplace=True, axis=1)

print("begin mmhc")
est = MmhcEstimator(data)
model = est.estimate()
print("finish mmhc")
print(model.edges())
