import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BDeuScore, BDsScore


print("begin")

data = pd.read_csv('training_kerala.csv')



print("loaded into pandas df")

data.drop(['latitude', 'longitude', 'time'], inplace=True, axis=1)

print("begin HCS")

est = HillClimbSearch(data)

best_model_bic = est.estimate(scoring_method=BicScore(data))
best_model_k2 = est.estimate(scoring_method=K2Score(data))
best_model_bdeu = est.estimate(scoring_method=BDeuScore(data))
best_model_bds = est.estimate(scoring_method=BDsScore(data))

print("finish HCS\n")

print(best_model_bic.nodes())
for edge in best_model_bic.edges():
    print(edge)


print(best_model_k2.nodes())
for edge in best_model_k2.edges():
    print(edge)


print(best_model_bdeu.nodes())
for edge in best_model_bdeu.edges():
    print(edge)


print(best_model_bds.nodes())
for edge in best_model_bds.edges():
    print(edge)
