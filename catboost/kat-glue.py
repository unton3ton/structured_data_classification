# https://github.com/mwaskom/seaborn-data/tree/master

# conda activate CAT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier


# Load the dataset
dataframe = pd.read_csv('glue.csv')
print(dataframe.head())

target = 'Encoder'


# Create the feature matrix (X) and target vector (y)
X = dataframe.drop(target, axis=1)
y = dataframe[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# specifying categorical features
categorical_features = ['Model', 'Task']
# create and train the CatBoostClassifier
model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
model.fit(X_train, y_train)



model.save_model('catboost_classification_glue.model')

model_name = CatBoostClassifier()      # parameters not required.
model_name.load_model('catboost_classification_glue.model')



# predicting accuracy
y_pred = model_name.predict(X_test)
# print(y_pred)
X_test['predicted'] = y_pred
print(X_test.head(11))

# saving the dataframe
X_test.to_csv('glue-predicted.csv')


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for glue')
plt.show()



importances = model_name.get_feature_importance()
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
 
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance for glue")
plt.show()



# Print the classification report
print("Classification Report for glue:")
print(classification_report(y_test, y_pred))


'''
          Model  Year   Task  Score    predicted
52  BiLSTM+ELMo  2018   QNLI   75.5         LSTM
58      RoBERTa  2019    RTE   88.2  Transformer
0         ERNIE  2019   CoLA   75.5  Transformer
44  BiLSTM+ELMo  2018   MNLI   67.2         LSTM
5   BiLSTM+CoVe  2017   CoLA   18.5         LSTM
36  BiLSTM+ELMo  2018    QQP   61.1         LSTM
16        ERNIE  2019   MRPC   93.9  Transformer
12  BiLSTM+ELMo  2018  SST-2   89.3         LSTM
25           T5  2019  STS-B   93.1  Transformer
61  BiLSTM+CoVe  2017    RTE   52.7         LSTM
56        ERNIE  2019    RTE   92.6  Transformer

Accuracy: 1.00

Classification Report for glue:
              precision    recall  f1-score   support

        LSTM       1.00      1.00      1.00         6
 Transformer       1.00      1.00      1.00         7

    accuracy                           1.00        13
   macro avg       1.00      1.00      1.00        13
weighted avg       1.00      1.00      1.00        13
'''