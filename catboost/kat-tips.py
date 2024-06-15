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
dataframe = pd.read_csv('tips.csv')
print(dataframe.head())

target = 'sex'


# Create the feature matrix (X) and target vector (y)
X = dataframe.drop(target, axis=1)
y = dataframe[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# specifying categorical features
categorical_features = ['smoker', 'day', 'time']
# create and train the CatBoostClassifier
model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
model.fit(X_train, y_train)



model.save_model('catboost_classification_tips.model')

model_name = CatBoostClassifier()      # parameters not required.
model_name.load_model('catboost_classification_tips.model')



# predicting accuracy
y_pred = model_name.predict(X_test)
# print(y_pred)
X_test['predicted'] = y_pred
print(X_test.head(11))

# saving the dataframe
X_test.to_csv('tips-predicted.csv')


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for tips')
plt.show()



importances = model_name.get_feature_importance()
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
 
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance for tips")
plt.show()



# Print the classification report
print("Classification Report for tips:")
print(classification_report(y_test, y_pred))

'''
     total_bill   tip smoker   day    time  size predicted
24        19.82  3.18     No   Sat  Dinner     2      Male
6          8.77  2.00     No   Sun  Dinner     2      Male
153       24.55  2.00     No   Sun  Dinner     4      Male
211       25.89  5.16    Yes   Sat  Dinner     4      Male
198       13.00  2.00    Yes  Thur   Lunch     2    Female
176       17.89  2.00    Yes   Sun  Dinner     2      Male
192       28.44  2.56    Yes  Thur   Lunch     2      Male
124       12.48  2.52     No  Thur   Lunch     2      Male
9         14.78  3.23     No   Sun  Dinner     2      Male
101       15.38  3.00    Yes   Fri  Dinner     2    Female
45        18.29  3.00     No   Sun  Dinner     2      Male

Accuracy: 0.63

Classification Report for tips:
              precision    recall  f1-score   support

      Female       0.56      0.26      0.36        19
        Male       0.65      0.87      0.74        30

    accuracy                           0.63        49
   macro avg       0.60      0.56      0.55        49
weighted avg       0.61      0.63      0.59        49
'''