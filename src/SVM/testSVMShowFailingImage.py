import os
import joblib
from matplotlib import pyplot as plt
import cv2
from skimage.transform import resize
from skimage.io import imread

modelPath = "joblibs/poly32x32.joblib"
scalerPath = 'joblibs/polyscaler32x32.joblib'
# Load the model
model = joblib.load(modelPath) 
classUnderTest = 4
path = f'validation/{classUnderTest}' 
images = []
fileNames = []
# Load the validation data
for img in os.listdir(path):
    try:
        img_array = imread(os.path.join(path, img))
        resized_img = resize(img_array, (32, 32), anti_aliasing=True)
        flattened_img = resized_img.flatten()
        images.append(flattened_img)
        fileNames.append(img)
    except Exception as e:
        print(f"Error processing {img}: {e}")
        continue

# Scale data
scaler = joblib.load(scalerPath)
x_test_normalised = scaler.transform(images)

print("predicting")
# Make predictions
prediction = model.predict(x_test_normalised)
print("done prediction")
# Check the shape of the predictions
print("Prediction shape:", prediction.shape)

i = 0
for i, pre in enumerate(prediction):
    if pre != classUnderTest:
        img_path = os.path.join(path, fileNames[i])
        img_display = cv2.imread(img_path)
        if img_display is not None:
            plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
            plt.title(f'Predicted Class: {pre}')
            plt.axis('off')  # Turn off axis
            plt.show()  # Display the image
        else:
            print(f"Error loading image for display: {fileNames[i]}")


# Convert predicted probabilities to class labels
if len(prediction.shape) == 1:
    y_pred_classes = prediction  # 1D output, no need for argmax
else:
    y_pred_classes = prediction.argmax(axis=1)  # Multi-class, use argmax for class labels
