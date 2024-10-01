import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from MLP import create_MLP
from sklearn import svm


df = pickle.load(open('pickles/kaydata.pickle', 'rb'))
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split our data into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 43, stratify = y, shuffle = True)

# Normalise the data.
scaler = MinMaxScaler()
x_train_normalised = scaler.fit_transform(x_train)
x_test_normalised = scaler.transform(x_test)

# These are the hyperparameters used for our MLP model as found from our grid search.
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

# Train the final MLP model.
MLP_model = create_MLP(
  hidden_layers = 2,
  nodes_per_layer = 128,
  activation = 'relu',
  learning_rate = 0.001,
  optimizer = 'adam'
)
MLP_model.fit(x_train_normalised, y_train, epochs = 20, batch_size = 32, verbose = 1)

# Save our final MLP model & test dataset.
joblib.dump(MLP_model, 'MLP_Model.joblib')
print('MLP training completed.')

# These are the hyperparameters used for our SVM model as found from our grid search.
# 'C': 1, 
# 'kernel': 'linear'

# Train the final SVM model.
# SVM_model = svm.SVC(
#   C = 1,
#   kernel = 'linear',
#   probability = True,
#   class_weight = 'balanced',
#   max_iter = 10000
# )
# SVM_model.fit(x_train_normalised, y_train)

# # Save our final SVM model & test dataset.
# joblib.dump(SVM_model, 'SVM_Model.joblib')
# print('SVM training completed.')

# # Save our testing dataset.
# joblib.dump((x_test_normalised, y_test), 'test_dataset.joblib')