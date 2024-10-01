import pickle
from sklearn.preprocessing import MinMaxScaler
from MLP import create_MLP
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

df = pickle.load(open('pickles/kaydata.pickle', 'rb'))
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split our data into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.70, random_state = 42, stratify = y, shuffle = True)

# Normalise the data.
scaler = MinMaxScaler()
x_train_normalised = scaler.fit_transform(x_train)
x_test_normalised = scaler.transform(x_test)

# We will use grid search to determine the best hyperparameters from a given set
# Including 3 fold validation we have 2916 different fits to adjust our hyperparameters.
param_grid = {
  'model__hidden_layers': [1, 2, 3],
  'model__nodes_per_layer': [32, 64, 128],
  'model__activation': ['relu', 'tanh', 'sigmoid'],
  'model__learning_rate': [0.001, 0.01, 0.1],
  'model__optimizer': ['adam', 'sgd'],
  'epochs': [10, 20, 30],
  'batch_size': [32, 64]
}

# param_grid = {
#   'model__hidden_layers': [2],
#   'model__nodes_per_layer': [64],
#   'model__activation': ['tanh'],
#   'model__learning_rate': [0.1],
#   'model__optimizer': ['sgd'],
#   'epochs': [30],
#   'batch_size': [64]
# }

# Create the KerasClassifier, with default parameters
model = KerasClassifier(
  model=create_MLP,
  verbose=1
)

# Perform grid search with StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=3)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=stratified_kfold, n_jobs=-1)
grid_result = grid.fit(x_train_normalised, y_train)

# Print out the test accuracy as well as training accuracy
test_accuracy = grid_result.best_estimator_.score(x_test_normalised, y_test)
print(f"Test Accuracy: {test_accuracy}")
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Test Accuracy: 0.9971461187214612
# Best: 0.9986666666666667 using {
# 'batch_size': 32, 
# 'epochs': 20, 
# 'model__activation': 'relu', 
# 'model__hidden_layers': 2, 
# 'model__learning_rate': 0.001, 
# 'model__nodes_per_layer': 128, 
# 'model__optimizer': 'adam'
# }

