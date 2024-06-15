# conda create --name CAT

# conda activate CAT

# pip install catboost --user 🐈

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier


# Load the dataset
dataframe = pd.read_csv('hearz.csv')
print(dataframe.head())

target = 'target'


# preprocessing data
# dataframe = dataframe.drop('variety', axis=1)
# print(dataframe.head())


# Create the feature matrix (X) and target vector (y)
X = dataframe.drop(target, axis=1)
y = dataframe[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# specifying categorical features
categorical_features = ['thal']
# create and train the CatBoostClassifier
model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
model.fit(X_train, y_train)



model.save_model('catboost_classification_hearz.model')

model_name = CatBoostClassifier()      # parameters not required.
model_name.load_model('catboost_classification_hearz.model')



# predicting accuracy
y_pred = model_name.predict(X_test)
# print(y_pred)
X_test['predicted'] = y_pred
print(X_test.head(11))

# saving the dataframe
X_test.to_csv('hearz-predicted.csv')


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for hearz')
plt.show()



importances = model_name.get_feature_importance()
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
 
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance for hearz")
plt.show()



# Print the classification report
print("Classification Report for hearz:")
print(classification_report(y_test, y_pred))

'''
Accuracy: 0.84

Classification Report for hearz:
              precision    recall  f1-score   support

           0       0.87      0.91      0.89        44
           1       0.73      0.65      0.69        17

    accuracy                           0.84        61
   macro avg       0.80      0.78      0.79        61
weighted avg       0.83      0.84      0.83        61
'''