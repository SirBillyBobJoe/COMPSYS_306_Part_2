import joblib
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


# Load the model
model = joblib.load("joblibs/standardised100x100.joblib") 

# Load the validation data
df = pickle.load(open('pickles/validationdata100x100.pickle', 'rb'))

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Normalise the data.
scaler = joblib.load('joblibs/scaler100x100.joblib')
x_test_normalised = scaler.transform(X)

print("predicting")
# Make predictions
prediction = model.predict(x_test_normalised)
print("done prediction")
# Check the shape of the predictions
print("Prediction shape:", prediction.shape)

# Convert predicted probabilities to class labels
if len(prediction.shape) == 1:
    y_pred_classes = prediction  # 1D output, no need for argmax
else:
    y_pred_classes = prediction.argmax(axis=1)  # Multi-class, use argmax for class labels

# Generate a classification report
print(f"\nClassification Report:\n{classification_report(y, y_pred_classes)}")
