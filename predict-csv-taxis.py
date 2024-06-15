import tensorflow as tf
from tensorflow.keras.models import load_model


savedModel = load_model('TaxiCsvModel.keras')
# print(f'savedModel.summary() = {savedModel.summary()}')

sample = { # cash
    "passengers": 1,
    "color": "yellow",
    "pickup_zone": "Upper West Side South",
    "dropoff_zone": "Upper West Side South",
    "pickup_borough": "Manhattan",
    "dropoff_borough": "Manhattan",
    "distance": 0.8,
    "fare": 5.0,
    "tip": 0.0,
    "tolls": 0.0,
    "total": 9.31,
} # This passengers had a 93.37% probability payment with a credit card.


# sample = { # credit card
#     "passengers": 1,
#     "color": "yellow",
#     "pickup_zone": "Lenox Hill West",
#     "dropoff_zone": "UN/Turtle Bay South",
#     "pickup_borough": "Manhattan",
#     "dropoff_borough": "Manhattan",
#     "distance": 1.60,
#     "fare": 7.0,
#     "tip": 2.15,
#     "tolls": 0.0,
#     "total": 12.95,
# } # This passengers had a 97.98% probability payment with a credit card.


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = savedModel.predict(input_dict)

print(
    f"This passengers had a {100 * predictions[0][0]:.2f}% probability "
    "payment with a credit card."
)