import joblib
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import classification_report

from python.SVM.SVM_Model_Kay import convert_to_grayscale

# Load the model
model = joblib.load("joblibs/SVM_Model_Kay.joblib") 

# Load the validation data
df = pickle.load(open('pickles/validationData.pickle', 'rb'))

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Apply grayscale conversion to all images in the dataset
x_grayscale = np.array([convert_to_grayscale(img) for img in X.values]) 

# Normalise the data.
scaler = joblib.load('joblibs/scaler.joblib')
x_test_normalised = scaler.transform(x_grayscale)

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
