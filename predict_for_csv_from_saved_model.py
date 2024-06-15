# conda activate iWM

import tensorflow as tf
from tensorflow.keras.models import load_model

savedModel = load_model('csvModel.keras')
print(f'savedModel.summary() = {savedModel.summary()}')

# with open('model.txt', 'w') as f:
#     savedModel.summary(print_fn=lambda x: f.write(x + '\n'))

"""
## Inference on new data with the end-to-end model

Now, we can use our inference model (which includes the `FeatureSpace`)
to make predictions based on dicts of raw features values, as follows:
"""

sample = {
    "age": 31,
    "sex": 1,
    "cp": 0,
    "trestbps": 118,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "normal",
}


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
# predictions = inference_model.predict(input_dict)
predictions = savedModel.predict(input_dict)

print(
    f"This particular patient had a {100 * predictions[0][0]:.2f}% probability "
    "of having a heart disease, as evaluated by our model."
)