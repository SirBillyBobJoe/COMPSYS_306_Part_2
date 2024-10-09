import joblib
import pickle
import tensorflow as tf
from sklearn.metrics import classification_report

# Load the model and scaler
model = joblib.load("./joblibs/SVM_Model.joblib")
scaler = joblib.load("./joblibs/SVM_Scaler.joblib")

# Load the validation data
df = pickle.load(open('pickles/validationData.pickle', 'rb'))

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Scale the features using the loaded scaler
X_scaled = scaler.transform(X)
print("predicting")
# Make predictions
prediction = model.predict(X_scaled)
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
