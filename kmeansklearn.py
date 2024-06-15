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



raw_data = pd.read_csv('classified_data.csv')
raw_data = pd.read_csv('classified_data.csv', index_col = 0)
print(raw_data.head())


scaler = StandardScaler()
scaler.fit(raw_data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(raw_data.drop('TARGET CLASS', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop('TARGET CLASS', axis=1).columns)

x = scaled_data
y = raw_data['TARGET CLASS']

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)


model = KNeighborsClassifier(n_neighbors = 1)

model.fit(x_training_data, y_training_data)

predictions = model.predict(x_test_data)
print(predictions)

print(classification_report(y_test_data, predictions))

print(confusion_matrix(y_test_data, predictions))
# Вывод:
# [[134  14]
# [ 12 140]]


'''
Выбор оптимального значения для K с помощью метода «Локтя»
'''

error_rates = []

for i in np.arange(1, 101):
    new_model = KNeighborsClassifier(n_neighbors = i)
    new_model.fit(x_training_data, y_training_data)
    new_predictions = new_model.predict(x_test_data)
    error_rates.append(np.mean(new_predictions != y_test_data))

plt.plot(error_rates)
plt.grid()
plt.show()
'''
Как видно из графика, мы достигаем минимальной частоты ошибок при значении K,
равном приблизительно 35. Это означает, что 35 является подходящим выбором для K,
который сочетает в себе простоту и точность предсказаний.
'''