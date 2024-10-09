import pickle
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

# Load dataset
data_dict = pickle.load(open('pickles/data32x32.pickle', 'rb'))
x = data_dict['data']
y = data_dict['target']

# Split our data into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42, stratify = y, shuffle = True)

# Normalise the data.
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)

# Including 3 folds for each candidate, we have 156 different fits to adjust our hyperparameters.
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ["poly"],
    'degree': [2, 3, 4, 5, 6],
    "gamma": [0.01, 0.1, 1, 10]
  }

# Create svc model and perform grid search
svc = svm.SVC(probability = True, class_weight='balanced', max_iter=10000)
model = GridSearchCV(svc, param_grid, verbose=3, cv=2, scoring='accuracy')
model.fit(x_train_std, y_train)

# Print out the test accuracy as well as training accuracy
test_accuracy = model.score(x_test_std, y_test)
print(f"Test Accuracy: {test_accuracy}")
print(f"Best: {model.best_score_} using {model.best_params_}")

# Test Accuracy: 1.0
# Best: 0.9990829310490398 using {'C': 0.001, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'}
# (.venv) kay@Kays-Laptop COMPSYS_306_Part_2 % 