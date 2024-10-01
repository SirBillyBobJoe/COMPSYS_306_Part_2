import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import numpy as np

df = pickle.load(open('pickles/dataset.pickle', 'rb'))
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Function to convert images to grayscale
def convert_to_grayscale(image_array):
  # Reshape the flattened image directly into (100, 100) for grayscale 
  image = Image.fromarray(image_array)  
  gray_image = image.convert("L")  # Convert to grayscale
  return np.array(gray_image).flatten()  # Flatten again if needed

# # Apply grayscale conversion to all images in the dataset
# x_grayscale = np.array([convert_to_grayscale(img) for img in x.values])

# # Split our data into training and test set.
# x_train, x_test, y_train, y_test = train_test_split(x_grayscale, y, test_size = 0.80, random_state = 42, stratify = y, shuffle = True)

# # Normalise the data.
# scaler = MinMaxScaler()
# x_train_normalised = scaler.fit_transform(x_train)
# x_test_normalised = scaler.transform(x_test)

# Including 3 folds for each candidate, we have 156 different fits to adjust our hyperparameters.
# param_grid = [
#   {
#     'C': [0.01, 0.1, 1, 10],
#     'kernel': ["linear"]
#   },
#   {
#     'C': [0.01, 0.1, 1, 10],
#     'kernel': ["poly"],
#     'degree': [2, 3, 4],
#     "gamma": [0.1, 1, 10]
#   },
#   {
#     'C': [0.01, 0.1, 1, 10],
#     'kernel': ["rbf"],
#     "gamma": [0.1, 1, 10] 
#   }
# ]

# param_grid = {
#     'C': [0.01],
#     'kernel': ["poly"],
#     'degree': [2],
#     "gamma": [0.1]
#   }

# # Create svc model and perform grid search
# svc = svm.SVC(probability = True, class_weight='balanced', max_iter=10000)
# model = GridSearchCV(svc, param_grid, verbose=3, cv=3, scoring='accuracy')
# model.fit(x_train_normalised, y_train)

# # Print out the test accuracy as well as training accuracy
# test_accuracy = model.score(x_test_normalised, y_test)
# print(f"Test Accuracy: {test_accuracy}")
# print(f"Best: {model.best_score_} using {model.best_params_}")

# Test Accuracy: 0.9831286073701346
# Best: 0.975133214920071 using {'C': 0.01, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'}