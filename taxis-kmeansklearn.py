# https://pythonru.com/uroki/sklearn-kmeans-i-knn
# https://gitlab.com/PythonRu/notebooks/-/blob/master/classified_data.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



raw_data = pd.read_csv('taxis.csv')
# raw_data = pd.read_csv('classified_data.csv', index_col = 0)
print(raw_data.head())

raw_data = raw_data.drop(['pickup', 'dropoff','color', 'pickup_zone', 'dropoff_zone', 'pickup_borough',
                        'dropoff_borough'], axis=1)
raw_data = raw_data.dropna()
print(raw_data.head())

scaler = StandardScaler()
scaler.fit(raw_data.drop('payment', axis=1))
scaled_features = scaler.transform(raw_data.drop('payment', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop('payment', axis=1).columns)

x = scaled_data
y = raw_data['payment']

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)


model = KNeighborsClassifier(n_neighbors = 1)

model.fit(x_training_data, y_training_data)

predictions = model.predict(x_test_data)
print(predictions)

print(classification_report(y_test_data, predictions))

print(confusion_matrix(y_test_data, predictions))
'''
              precision    recall  f1-score   support

        cash       0.85      0.88      0.87       544
 credit card       0.95      0.94      0.95      1373

    accuracy                           0.92      1917
   macro avg       0.90      0.91      0.91      1917
weighted avg       0.92      0.92      0.92      1917

[[ 479   65]
 [  83 1290]]

'''