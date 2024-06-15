# conda create --name CAT

# conda activate CAT

# pip install catboost --user
# pip install seaborn scikit-learn

# pip freeze > katrequireements.txt

# conda deactivate

'''
📌 Кто такой catBoost? 🐈 CatBoost означает «категорическое» повышение.

CatBoost — это библиотека градиентного бустинга, созданная 🌐ндексом.
Прогнозы делаются на основе ансамбля слабых обучающих алгоритмов, а именно небрежных деревьев.
Вот несколько преимущества использования этой библиотеки: 

➖ позволяет использовать категориальные признаки без предварительной обработки  
➖ дает отличные результаты с параметрами по умолчанию
➖ под капотом умеет обрабатывать пропущенные значения
➖ можно использовать и для регрессии, и для класссификации
'''

'''
Она использует небрежные (oblivious) деревья решений, чтобы вырастить сбалансированное дерево.
Одни и те же функции используются для создания левых и правых разделений (split) на каждом уровне дерева.

В плане простоты использования и легкости входа для новичков, пожалуй является топ-1 библиотекой для табличных данных
и вот почему:

⏩Принимает категориальные фичи сразу без всякой предварительной обработки.

⏩Чтобы перенести обучение с CPU на GPU достаточно поменять значение 1 параметра, без установки доп.пакетов или специальных версий, как в других библиотеках

⏩Даже с дефолтными параметрами выдает хорошую точность модели. Основные параметры не константные, а подбираются самой библиотекой, в зависимости от размера входных данных.

⏩Может принимать текстовые признаки, эмбеддинги, временные признаки.

⏩Без дополнительных манипуляций и оберток встраивается в стандартные пайплайны (например, sklearn).

⏩Идет в комплекте с "батарейками": feature_selection, object_selection, cross_validation, grid_search и пр.
'''

'''
Бустинг – это ансамблевый метод машинного обучения, целью которого является объединение нескольких слабых моделей
предсказания для создания одной сильной. Слабая модель – это такая, которая выполняет предсказания немного лучше,
чем наугад, в то время как сильная модель обладает высокой предсказательной способностью. Цель бустинга – улучшить
точность предсказаний.

Бустинг работает путём последовательного добавления моделей в ансамбль. Каждая следующая модель строится таким образом,
чтобы исправлять ошибки, сделанные предыдущими моделями. Это достигается путём фокусировки на наиболее проблемных данных,
которые были неверно классифицированы или предсказаны ранее.

Одной из основных фич бустинга является динамическое взвешивание обучающих данных. После каждого этапа обучения модели в
ансамбле, данные, на которых были допущены ошибки, получают больший вес. Это означает, что последующие модели уделяют
больше внимания именно этим трудным случаям.

Когда используются решающие деревья, каждое последующее дерево строится с учетом ошибок, сделанных предыдущими деревьями.
Новые деревья учатся на ошибках, улучшая общую точность ансамбля.

Несмотря на свою мощь, бустинг может быть склонен к переобучению, особенно если в ансамбле слишком много моделей или они
слишком сложные. Для контроля переобучения к примеру ранняя остановка (early stopping).
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier




# Load the Titanic dataset
# titanic = sns.load_dataset('titanic') # https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv
# target = 'survived'

titanic = pd.read_csv('titanic.csv')
print(titanic.head())

target = 'survived'



# preprocessing data
 
# filling missing value in deck column with a new category: Unknown
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown']
titanic['deck'] = pd.Categorical(
    titanic['deck'], categories=categories, ordered=True)
titanic['deck'] = titanic['deck'].fillna('Unknown')
 
# filling missing value in age column using mean imputation
age_mean = titanic['age'].fillna(0).mean()
titanic['age'] = titanic['age'].fillna(age_mean)
 
# droping missing values in embark as there are only 2
titanic = titanic.dropna()
 
# droping alive column to make the problem more challenging
titanic = titanic.drop('alive', axis=1)



# Create the feature matrix (X) and target vector (y)
X = titanic.drop(target, axis=1)
y = titanic[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# specifying categorical features
categorical_features = ['sex', 'pclass', 'sibsp', 'parch', 'embarked',
                        'class', 'who', 'adult_male', 'embark_town', 'alone', 'deck']
# create and train the CatBoostClassifier
model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
# model.fit(X_train, y_train)



# model.save_model('catboost_classification_titanic.model')

model_name = CatBoostClassifier()      # parameters not required.
model_name.load_model('catboost_classification_titanic.model')



# predicting accuracy
y_pred = model_name.predict(X_test)
# print(y_pred)
X_test['predicted'] = y_pred
print(X_test.head(11))

# saving the dataframe
X_test.to_csv('titanic-predicted.csv')


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



importances = model_name.get_feature_importance()
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
 
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance")
plt.show()



# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


'''
Accuracy: 0.80

Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.87      0.84       109
           1       0.77      0.70      0.73        69

    accuracy                           0.80       178
   macro avg       0.80      0.78      0.79       178
weighted avg       0.80      0.80      0.80       178
'''