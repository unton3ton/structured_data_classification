# https://github.com/mwaskom/seaborn-data/tree/master
# https://github.com/mwaskom/seaborn-data/blob/master/taxis.csv

# conda activate CAT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier


# Load the dataset
dataframe = pd.read_csv('taxis.csv')
print(dataframe.head())
print(dataframe.shape)
print(dataframe.info())

'''
                pickup              dropoff  passengers  distance  fare   tip  tolls  total   color      payment            pickup_zone           dropoff_zone pickup_borough dropoff_borough
0  2019-03-23 20:21:09  2019-03-23 20:27:24           1      1.60   7.0  2.15    0.0  12.95  yellow  credit card        Lenox Hill West    UN/Turtle Bay South      Manhattan       Manhattan
1  2019-03-04 16:11:55  2019-03-04 16:19:00           1      0.79   5.0  0.00    0.0   9.30  yellow         cash  Upper West Side South  Upper West Side South      Manhattan       Manhattan
2  2019-03-27 17:53:01  2019-03-27 18:00:25           1      1.37   7.5  2.36    0.0  14.16  yellow  credit card          Alphabet City           West Village      Manhattan       Manhattan
3  2019-03-10 01:23:59  2019-03-10 01:49:51           1      7.70  27.0  6.15    0.0  36.95  yellow  credit card              Hudson Sq         Yorkville West      Manhattan       Manhattan
4  2019-03-30 13:27:42  2019-03-30 13:37:14           3      2.16   9.0  1.10    0.0  13.40  yellow  credit card           Midtown East         Yorkville West      Manhattan       Manhattan
(6433, 14)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6433 entries, 0 to 6432
Data columns (total 14 columns):
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   pickup           6433 non-null   object
 1   dropoff          6433 non-null   object
 2   passengers       6433 non-null   int64
 3   distance         6433 non-null   float64
 4   fare             6433 non-null   float64
 5   tip              6433 non-null   float64
 6   tolls            6433 non-null   float64
 7   total            6433 non-null   float64
 8   color            6433 non-null   object
 9   payment          6389 non-null   object
 10  pickup_zone      6407 non-null   object
 11  dropoff_zone     6388 non-null   object
 12  pickup_borough   6407 non-null   object
 13  dropoff_borough  6388 non-null   object
dtypes: float64(5), int64(1), object(8)
memory usage: 703.7+ KB
'''

target = 'payment'

# preprocessing data: cat_features must be integer or string, real number values and NaN values should be converted to string.
dataframe = dataframe.dropna()
# dataframe = dataframe.drop(['pickup', 'dropoff'], axis=1)
dataframe['dropoff'] = dataframe['dropoff'].astype(str)
dataframe['pickup'] = dataframe['pickup'].astype(str)


# Create the feature matrix (X) and target vector (y)
X = dataframe.drop(target, axis=1)
y = dataframe[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# specifying categorical features
categorical_features = ['pickup', 'dropoff','color', 'pickup_zone', 'dropoff_zone', 'pickup_borough',
                        'dropoff_borough']
# create and train the CatBoostClassifier
model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
model.fit(X_train, y_train)

'''
0:      learn: 0.6233187        total: 176ms    remaining: 17.5s
1:      learn: 0.5582761        total: 182ms    remaining: 8.92s
2:      learn: 0.5070269        total: 196ms    remaining: 6.33s
3:      learn: 0.4623906        total: 209ms    remaining: 5.03s
4:      learn: 0.4268648        total: 224ms    remaining: 4.26s
5:      learn: 0.3923146        total: 236ms    remaining: 3.69s
6:      learn: 0.3627496        total: 244ms    remaining: 3.24s
7:      learn: 0.3379926        total: 256ms    remaining: 2.95s
8:      learn: 0.3163899        total: 271ms    remaining: 2.75s
9:      learn: 0.2968414        total: 279ms    remaining: 2.51s
10:     learn: 0.2796024        total: 291ms    remaining: 2.36s
11:     learn: 0.2645461        total: 305ms    remaining: 2.24s
12:     learn: 0.2509147        total: 319ms    remaining: 2.13s
13:     learn: 0.2390644        total: 332ms    remaining: 2.04s
14:     learn: 0.2283068        total: 346ms    remaining: 1.96s
15:     learn: 0.2187057        total: 360ms    remaining: 1.89s
16:     learn: 0.2098948        total: 374ms    remaining: 1.82s
17:     learn: 0.2025366        total: 388ms    remaining: 1.76s
18:     learn: 0.1956326        total: 401ms    remaining: 1.71s
19:     learn: 0.1889538        total: 414ms    remaining: 1.66s
20:     learn: 0.1831394        total: 428ms    remaining: 1.61s
21:     learn: 0.1776855        total: 441ms    remaining: 1.56s
22:     learn: 0.1730124        total: 455ms    remaining: 1.52s
23:     learn: 0.1686998        total: 469ms    remaining: 1.48s
24:     learn: 0.1649280        total: 482ms    remaining: 1.45s
25:     learn: 0.1614438        total: 497ms    remaining: 1.42s
26:     learn: 0.1581258        total: 518ms    remaining: 1.4s
27:     learn: 0.1551839        total: 533ms    remaining: 1.37s
28:     learn: 0.1524781        total: 546ms    remaining: 1.34s
29:     learn: 0.1499819        total: 560ms    remaining: 1.31s
30:     learn: 0.1476731        total: 575ms    remaining: 1.28s
31:     learn: 0.1458057        total: 588ms    remaining: 1.25s
32:     learn: 0.1439919        total: 602ms    remaining: 1.22s
33:     learn: 0.1420310        total: 616ms    remaining: 1.2s
34:     learn: 0.1404973        total: 630ms    remaining: 1.17s
35:     learn: 0.1389005        total: 644ms    remaining: 1.14s
36:     learn: 0.1371300        total: 658ms    remaining: 1.12s
37:     learn: 0.1357909        total: 672ms    remaining: 1.09s
38:     learn: 0.1345599        total: 685ms    remaining: 1.07s
39:     learn: 0.1338893        total: 699ms    remaining: 1.05s
40:     learn: 0.1328265        total: 712ms    remaining: 1.02s
41:     learn: 0.1316869        total: 726ms    remaining: 1s
42:     learn: 0.1307625        total: 740ms    remaining: 981ms
43:     learn: 0.1296670        total: 754ms    remaining: 960ms
44:     learn: 0.1287822        total: 763ms    remaining: 932ms
45:     learn: 0.1279533        total: 776ms    remaining: 911ms
46:     learn: 0.1271221        total: 790ms    remaining: 891ms
47:     learn: 0.1263117        total: 804ms    remaining: 871ms
48:     learn: 0.1259618        total: 817ms    remaining: 850ms
49:     learn: 0.1254444        total: 831ms    remaining: 831ms
50:     learn: 0.1246826        total: 845ms    remaining: 812ms
51:     learn: 0.1243692        total: 859ms    remaining: 793ms
52:     learn: 0.1237797        total: 873ms    remaining: 774ms
53:     learn: 0.1232281        total: 887ms    remaining: 756ms
54:     learn: 0.1225791        total: 901ms    remaining: 737ms
55:     learn: 0.1219648        total: 914ms    remaining: 719ms
56:     learn: 0.1219299        total: 919ms    remaining: 693ms
57:     learn: 0.1215192        total: 932ms    remaining: 675ms
58:     learn: 0.1213138        total: 946ms    remaining: 657ms
59:     learn: 0.1212711        total: 962ms    remaining: 641ms
60:     learn: 0.1207654        total: 977ms    remaining: 624ms
61:     learn: 0.1206010        total: 991ms    remaining: 607ms
62:     learn: 0.1200860        total: 1s       remaining: 591ms
63:     learn: 0.1199512        total: 1.02s    remaining: 574ms
64:     learn: 0.1194240        total: 1.03s    remaining: 558ms
65:     learn: 0.1189549        total: 1.05s    remaining: 541ms
66:     learn: 0.1188439        total: 1.06s    remaining: 522ms
67:     learn: 0.1187920        total: 1.06s    remaining: 501ms
68:     learn: 0.1187825        total: 1.07s    remaining: 480ms
69:     learn: 0.1185256        total: 1.08s    remaining: 464ms
70:     learn: 0.1185190        total: 1.09s    remaining: 445ms
71:     learn: 0.1181260        total: 1.1s     remaining: 428ms
72:     learn: 0.1178732        total: 1.12s    remaining: 413ms
73:     learn: 0.1175964        total: 1.13s    remaining: 397ms
74:     learn: 0.1172277        total: 1.14s    remaining: 381ms
75:     learn: 0.1168594        total: 1.16s    remaining: 366ms
76:     learn: 0.1166237        total: 1.17s    remaining: 350ms
77:     learn: 0.1163408        total: 1.18s    remaining: 334ms
78:     learn: 0.1160788        total: 1.2s     remaining: 318ms
79:     learn: 0.1158453        total: 1.21s    remaining: 303ms
80:     learn: 0.1155340        total: 1.23s    remaining: 288ms
81:     learn: 0.1152641        total: 1.24s    remaining: 272ms
82:     learn: 0.1150903        total: 1.25s    remaining: 257ms
83:     learn: 0.1147869        total: 1.27s    remaining: 242ms
84:     learn: 0.1145184        total: 1.28s    remaining: 226ms
85:     learn: 0.1142394        total: 1.29s    remaining: 211ms
86:     learn: 0.1139451        total: 1.31s    remaining: 196ms
87:     learn: 0.1136813        total: 1.32s    remaining: 180ms
88:     learn: 0.1134232        total: 1.34s    remaining: 165ms
89:     learn: 0.1133195        total: 1.35s    remaining: 150ms
90:     learn: 0.1130514        total: 1.36s    remaining: 135ms
91:     learn: 0.1129326        total: 1.38s    remaining: 120ms
92:     learn: 0.1127142        total: 1.39s    remaining: 105ms
93:     learn: 0.1127105        total: 1.4s     remaining: 89.1ms
94:     learn: 0.1125976        total: 1.41s    remaining: 74.2ms
95:     learn: 0.1123823        total: 1.42s    remaining: 59.3ms
96:     learn: 0.1123746        total: 1.43s    remaining: 44.2ms
97:     learn: 0.1123363        total: 1.44s    remaining: 29.4ms
98:     learn: 0.1122176        total: 1.45s    remaining: 14.7ms
99:     learn: 0.1120782        total: 1.47s    remaining: 0us
'''

model.save_model('catboost_classification_taxis.model')

model_name = CatBoostClassifier()      # parameters not required.
model_name.load_model('catboost_classification_taxis.model')



# predicting accuracy
y_pred = model_name.predict(X_test)
# print(y_pred)
X_test['predicted'] = y_pred
print(X_test.head(11))

# saving the dataframe
X_test.to_csv('taxis-predicted.csv')


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for taxis')
plt.show()



importances = model_name.get_feature_importance()
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
 
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance for taxis")
plt.show()



# Print the classification report
print("Classification Report for taxis:")
print(classification_report(y_test, y_pred))


'''
                   pickup              dropoff  passengers  distance   fare   tip  ...   color              pickup_zone                    dropoff_zone pickup_borough dropoff_borough    predicted
742   2019-03-05 09:52:36  2019-03-05 10:03:47           1      1.32   8.50  0.00  ...  yellow  Greenwich Village North                     Murray Hill      Manhattan       Manhattan         cash
4824  2019-03-17 13:16:13  2019-03-17 13:40:32           1      2.90  17.00  0.00  ...  yellow             Clinton East                 Lenox Hill East      Manhattan       Manhattan         cash
3108  2019-03-14 01:33:26  2019-03-14 01:45:58           1      4.17  14.50  3.66  ...  yellow          Lower East Side                  Midtown Center      Manhattan       Manhattan  credit card
4985  2019-03-05 11:41:34  2019-03-05 12:12:45           1      4.03  21.50  2.50  ...  yellow             Clinton West             Little Italy/NoLiTa      Manhattan       Manhattan  credit card
219   2019-03-08 18:08:03  2019-03-08 18:15:42           1      1.40   7.00  2.25  ...  yellow                 Kips Bay                   Alphabet City      Manhattan       Manhattan  credit card
4154  2019-03-11 18:57:33  2019-03-11 19:06:21           1      1.10   7.50  2.00  ...  yellow             Midtown East    Penn Station/Madison Sq West      Manhattan       Manhattan  credit card
2280  2019-03-17 12:10:05  2019-03-17 12:15:02           1      1.00   6.00  0.70  ...  yellow          Lenox Hill East                  Yorkville East      Manhattan       Manhattan  credit card
5456  2019-03-12 21:11:03  2019-03-12 21:41:36           1     15.78  42.82  0.00  ...   green                  Maspeth                     Marble Hill         Queens       Manhattan  credit card
1684  2019-03-28 19:53:03  2019-03-28 20:04:10           6      2.39  10.50  2.96  ...  yellow           Midtown Center  Long Island City/Hunters Point      Manhattan          Queens  credit card
4889  2019-03-28 12:24:30  2019-03-28 12:28:44           2      0.95   5.00  1.66  ...  yellow    Upper West Side North                Manhattan Valley      Manhattan       Manhattan  credit card
5395  2019-03-02 19:36:34  2019-03-02 20:05:06           1      3.37  18.50  3.00  ...  yellow    Upper East Side South                        Union Sq      Manhattan       Manhattan  credit card

[11 rows x 14 columns]

Accuracy: 0.97

Classification Report for taxis:
              precision    recall  f1-score   support

        cash       0.92      0.98      0.95       365
 credit card       0.99      0.97      0.98       904

    accuracy                           0.97      1269
   macro avg       0.95      0.97      0.96      1269
weighted avg       0.97      0.97      0.97      1269
'''