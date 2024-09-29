import pickle
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from python.SVM.SVM_Model_Kay import convert_to_grayscale


df = pickle.load(open('pickles/dataset.pickle', 'rb'))
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Apply grayscale conversion to all images in the dataset
x_grayscale = np.array([convert_to_grayscale(img) for img in x.values])

# Split our data into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x_grayscale, y, test_size = 0.05, random_state = 43, stratify = y, shuffle = True)

# Normalise the data.
scaler = MinMaxScaler()
x_train_normalised = scaler.fit_transform(x_train)
x_test_normalised = scaler.transform(x_test)
joblib.dump(scaler, 'scaler.joblib') 

# These are the hyperparameters used for our SVM model as found from our grid search.
# 'C': [0.01],
# 'kernel': ["poly"],
# 'degree': [2],
# "gamma": [0.1]

# Train the final SVM model.
model = svm.SVC(
  C = 0.01,
  kernel = 'linear',
  probability = True,
  class_weight = 'balanced',
  max_iter = 10000,
  verbose=True
)
model.fit(x_train_normalised, y_train)

# Save our final SVM model & test dataset.
joblib.dump(model, 'joblibs/SVM_Model_Kay.joblib')
print('SVM training completed.')

# Fitting 3 folds for each of 1 candidates, totalling 3 fits
# [CV 1/3] END C=0.01, degree=2, gamma=0.1, kernel=poly;, score=0.984 total time= 1.0min
# [CV 2/3] END C=0.01, degree=2, gamma=0.1, kernel=poly;, score=0.973 total time= 1.1min
# [CV 3/3] END C=0.01, degree=2, gamma=0.1, kernel=poly;, score=0.968 total time= 1.1min
# Test Accuracy: 0.9831286073701346
# Best: 0.975133214920071 using {'C': 0.01, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'}
