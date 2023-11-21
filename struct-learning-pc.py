import pandas as pd
from pgmpy.estimators import PC


print("begin")

data = pd.read_csv('training_kerala.csv')
print("loaded into pandas df")
data.drop(['latitude', 'longitude', 'time'], inplace=True, axis=1)

print("begin pc")
est = PC(data)

model_chi = est.estimate(ci_test='chi_square')
print(len(model_chi.edges()))

model_gsq, _ = est.estimate(ci_test='g_sq', return_type='skeleton')
print(len(model_gsq.edges()))
print("finish pc")

print(model_gsq.edges())
