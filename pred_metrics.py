from pgmpy.models import BayesianNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

print("parameter est with mmhc")
train_data = pd.read_csv('training_kerala.csv')
test_data = pd.read_csv('testing_kerala.csv')


nodes = ['msl', 'tcc', 'tp', 'r', 't']
edges_hcbic = [('msl', 't'), ('tcc', 'r'), ('tcc', 'msl'), ('tcc', 'tp'), ('tcc', 't'), ('tp', 'r'), ('tp', 'msl'), ('r', 't'), ('r', 'msl')]
edges_hck2 = [('tcc', 'msl'), ('tp', 'msl'), ('tp', 'tcc'), ('tp', 'r'), ('tp', 't'), ('r', 'tcc'), ('r', 'msl'), ('r', 't'), ('t', 'msl'), ('t', 'tcc')]
edges_mmhc = [('msl', 't'), ('tcc', 'msl'), ('tcc', 'tp'), ('tcc', 't'), ('tp', 'msl'), ('tp', 't'), ('r', 't'), ('r', 'tcc'), ('r', 'msl'), ('r', 'tp')]
edges_pc = [('msl', 'tcc'), ('msl', 'tp'), ('msl', 'r'), ('msl', 't'), ('tcc', 'tp'), ('tcc', 'r'), ('tcc', 't'), ('tp', 'r'), ('tp', 't'), ('r', 't')]
edges_hcbdeu = [('msl', 't'), ('tcc', 'r'), ('tcc', 'tp'), ('tcc', 'msl'), ('tcc', 't'), ('tp', 'msl'), ('tp', 't'), ('r', 't'), ('r', 'tp'), ('r', 'msl')]
train_data.drop(['latitude', 'longitude', 'time'], inplace=True, axis=1)
test_data.drop(['latitude', 'longitude', 'time'], inplace=True, axis=1)

model = BayesianNetwork()
model.add_nodes_from(nodes)
model.add_edges_from(edges_mmhc)
model.fit(train_data)

actual = test_data['tp'] 
test_data.drop(['tp'], inplace=True, axis=1)


print("making predictions")

predicted = model.predict(test_data, stochastic=True)

predicted = predicted["tp"]
print("here")
print(set(actual) - set(predicted))
print(set(predicted) - set(actual))
print(actual.dtype)
print(predicted.dtypes)

accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted)
recall = recall_score(actual, predicted)
f1 = f1_score(actual, predicted)
conf_matrix = confusion_matrix(actual, predicted)
tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("true pos", tp)
print("true neg", tn)
print("false pos", fp)
print("false neg", fn)

print("Precision:", precision)
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("F1-Score:", f1)


false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)

# Plotting the ROC curve
plt.figure()
plt.plot(false_positive_rate, true_positive_rate, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Plotting the random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()








