import joblib
import numpy as np
from skimage.io import imread
import pandas as pd

# Load the model and scaler
model = joblib.load("./joblibs/SVM_Model.joblib")
scaler = joblib.load("./joblibs/scaler.joblib")

# Load and preprocess the image
image = imread("validation/4_4.jpg")

# Flatten the image
image_flattened = image.flatten()

# Reshape the image to match the model input
image_reshaped = image_flattened.reshape(1, -1)

# Create a DataFrame (optional, if you need it for further processing)
df = pd.DataFrame(image_reshaped)

# Scale the image using the loaded scaler
image_scaled = scaler.transform(df)

# Now you can make predictions
prediction = model.predict(image_scaled)
print("Predicted class:", prediction)
