# https://keras.io/examples/structured_data/structured_data_classification_with_feature_space/
# https://github.com/keras-team/keras-io/blob/master/examples/structured_data/structured_data_classification_with_feature_space.py

import os, numpy

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace


"""
## Preparing the data

Let's download the data and load it into a Pandas dataframe:
"""

file_url = "taxis.csv" # "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)

dataframe = dataframe.drop(['pickup', 'dropoff'], axis=1)
dataframe = dataframe.dropna()
print(dataframe.head())

# # замена строкового значения на целочисленное для бинарной классификации
# dataframe['target'] = dataframe.loc[dataframe['payment'] =='credit card', 'target'] = 1
# dataframe['target'] = dataframe.loc[dataframe['payment'] =='cash', 'target'] = 0

# creating a dict file  
pay = {'credit card': 1,'cash': 0} 
dataframe.payment = [pay[item] for item in dataframe.payment]


"""
The dataset includes 303 samples with 14 columns per sample
(13 features, plus the target label):
"""

print(f"dataframe.shape = {dataframe.shape}")

"""
Here's a preview of a few samples:
"""

print(dataframe.head(20))
print(dataframe.info())

#    passengers  distance  fare   tip  tolls  total   color      payment            pickup_zone           dropoff_zone pickup_borough dropoff_borough
# 0           1      1.60   7.0  2.15    0.0  12.95  yellow  credit card        Lenox Hill West    UN/Turtle Bay South      Manhattan       Manhattan
# 1           1      0.79   5.0  0.00    0.0   9.30  yellow         cash  Upper West Side South  Upper West Side South      Manhattan       Manhattan
# 2           1      1.37   7.5  2.36    0.0  14.16  yellow  credit card          Alphabet City           West Village      Manhattan       Manhattan
# 3           1      7.70  27.0  6.15    0.0  36.95  yellow  credit card              Hudson Sq         Yorkville West      Manhattan       Manhattan
# 4           3      2.16   9.0  1.10    0.0  13.40  yellow  credit card           Midtown East         Yorkville West      Manhattan       Manhattan
# dataframe.shape = (6341, 12)
#     passengers  distance  fare   tip  tolls  total   color  payment                   pickup_zone                   dropoff_zone pickup_borough dropoff_borough
# 0            1      1.60   7.0  2.15    0.0  12.95  yellow        1               Lenox Hill West            UN/Turtle Bay South      Manhattan       Manhattan
# 1            1      0.79   5.0  0.00    0.0   9.30  yellow        0         Upper West Side South          Upper West Side South      Manhattan       Manhattan
# 2            1      1.37   7.5  2.36    0.0  14.16  yellow        1                 Alphabet City                   West Village      Manhattan       Manhattan
# 3            1      7.70  27.0  6.15    0.0  36.95  yellow        1                     Hudson Sq                 Yorkville West      Manhattan       Manhattan
# 4            3      2.16   9.0  1.10    0.0  13.40  yellow        1                  Midtown East                 Yorkville West      Manhattan       Manhattan
# 5            1      0.49   7.5  2.16    0.0  12.96  yellow        1     Times Sq/Theatre District                   Midtown East      Manhattan       Manhattan
# 6            1      3.65  13.0  2.00    0.0  18.80  yellow        1             Battery Park City        Two Bridges/Seward Park      Manhattan       Manhattan
# 8            1      3.63  15.0  1.00    0.0  19.30  yellow        1             East Harlem South                 Midtown Center      Manhattan       Manhattan
# 9            1      1.52   8.0  1.00    0.0  13.30  yellow        1           Lincoln Square East                   Central Park      Manhattan       Manhattan
# 10           1      3.90  17.0  0.00    0.0  17.80  yellow        0             LaGuardia Airport                        Astoria         Queens          Queens
# 11           1      1.53   6.5  2.16    0.0  12.96  yellow        1         Upper West Side South               Manhattan Valley      Manhattan       Manhattan
# 12           1      1.05   6.5  1.00    0.0  11.30  yellow        1                   Murray Hill                 Midtown Center      Manhattan       Manhattan
# 13           1      1.75  10.5  0.00    0.0  13.80  yellow        0           Lincoln Square West      Times Sq/Theatre District      Manhattan       Manhattan
# 14           0      2.90  11.5  0.00    0.0  14.80  yellow        0      Financial District North        Two Bridges/Seward Park      Manhattan       Manhattan
# 15           3      2.09  13.5  0.00    0.0  16.80  yellow        0         Upper West Side North                   Clinton East      Manhattan       Manhattan
# 16           1      2.12  13.0  0.00    0.0  16.30  yellow        0                  East Chelsea  Meatpacking/West Village West      Manhattan       Manhattan
# 17           1      2.60  10.5  2.00    0.0  16.30  yellow        1                Midtown Center              East Harlem South      Manhattan       Manhattan
# 18           1      2.18   9.5  1.92    0.0  14.72  yellow        1                      Gramercy                 Midtown Center      Manhattan       Manhattan
# 19           6      1.08   6.5  1.08    0.0  11.38  yellow        1                  East Chelsea                   East Chelsea      Manhattan       Manhattan
# 20           1      1.07   6.5  1.54    0.0  11.84  yellow        1  Penn Station/Madison Sq West                       Kips Bay      Manhattan       Manhattan


"""
The last column, "target", indicates whether the patient
has a heart disease (1) or not (0).

Let's split the data into a training and validation set:
"""

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

"""
Let's generate `tf.data.Dataset` objects for each dataframe:
"""


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("payment")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)


for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

"""
Let's batch the datasets:
"""

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

"""
## Configuring a `FeatureSpace`

To configure how each feature should be preprocessed,
we instantiate a `keras.utils.FeatureSpace`, and we
pass to it a dictionary that maps the name of our features
to a string that describes the feature type.

We have a few "integer categorical" features such as `"FBS"`,
one "string categorical" feature (`"thal"`),
and a few numerical features, which we'd like to normalize
-- except `"age"`, which we'd like to discretize into
a number of bins.

We also use the `crosses` argument
to capture *feature interactions* for some categorical
features, that is to say, create additional features
that represent value co-occurrences for these categorical features.
You can compute feature crosses like this for arbitrary sets of
categorical features -- not just tuples of two features.
Because the resulting co-occurences are hashed
into a fixed-sized vector, you don't need to worry about whether
the co-occurence space is too large.
"""

feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "passengers": "integer_categorical",
        
        # Categorical feature encoded as string
        "color": "string_categorical",
        "pickup_zone": "string_categorical",
        "dropoff_zone": "string_categorical",
        "pickup_borough": "string_categorical",
        "dropoff_borough": "string_categorical",
        # Numerical features to discretize
        "distance": "float_discretized",
        "fare": "float_discretized",
        "tip": "float_discretized",
        "tolls": "float_discretized",
        "total": "float_discretized",
    },
    # We create additional features by hashing
    # value co-occurrences for the
    # following groups of categorical features.
    crosses=[("pickup_borough", "dropoff_borough"), ("pickup_zone", "tip")],
    # The hashing space for these co-occurrences
    # wil be 32-dimensional.
    crossing_dim=32,
    # Our utility will one-hot encode all categorical
    # features and concat all features into a single
    # vector (one vector per sample).
    output_mode="concat",
)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)

"""
Let's create a training and validation dataset of preprocessed batches:
"""

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

"""
## Build a model

Time to build a model -- or rather two models:

- A training model that expects preprocessed features (one sample = one vector)
- An inference model that expects raw features (one sample = dict of raw feature values)
"""

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(32, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)

"""
## Train the model

Let's train our model for 50 epochs. Note that feature preprocessing is happening
as part of the tf.data pipeline, not as part of the model.
"""

training_model.fit(
    preprocessed_train_ds,
    epochs=20,
    validation_data=preprocessed_val_ds,
    verbose=2,
)

"""
We quickly get to 80% validation accuracy.
"""


# save model
# inference_model.save('csvModel.h5')
inference_model.save('TaxiCsvModel.keras')
print('Model Saved!')
 
# # load model
from tensorflow.keras.models import load_model
savedModel=load_model('TaxiCsvModel.keras')
print(f'savedModel.summary() = {savedModel.summary()}')


"""
## Inference on new data with the end-to-end model

Now, we can use our inference model (which includes the `FeatureSpace`)
to make predictions based on dicts of raw features values, as follows:
"""

sample = {
    "passengers": 1,
    "color": "yellow",
    "pickup_zone": "Lenox Hill West",
    "dropoff_zone": "UN/Turtle Bay South",
    "pickup_borough": "Manhattan",
    "dropoff_borough": "Manhattan",
    "distance": 1.60,
    "fare": 7.0,
    "tip": 2.15,
    "tolls": 0.0,
    "total": 12.95,
}


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = inference_model.predict(input_dict)
# predictions = savedModel.predict(input_dict)

print(
    f"This passengers had a {100 * predictions[0][0]:.2f}% probability "
    "payment with a credit card."
)


predictions = savedModel.predict(input_dict)

print(
    f"This passengers had a {100 * predictions[0][0]:.2f}% probability "
    "payment with a credit card."
)

# This passengers had a 100.00% probability payment with a credit card.

# This passengers had a 97.98% probability payment with a credit card.