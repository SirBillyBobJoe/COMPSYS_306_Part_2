import joblib
import pickle
from sklearn.metrics import classification_report

modelPath = "joblibs/svm_model.joblib"
scalarPath = 'joblibs/scaler.joblib'
validationPath = 'pickles/validationData.pickle'

# Load the model
model = joblib.load(modelPath) 

# Load the validation data
data_dict  = pickle.load(open(validationPath, 'rb'))

X = data_dict['data']
y = data_dict['target']

# Normalise the data.
scaler = joblib.load(scalarPath)
x_test_standardised = scaler.transform(X)

print("predicting")
# Make predictions
prediction = model.predict(x_test_standardised)
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
