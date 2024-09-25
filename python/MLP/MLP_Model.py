import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scikeras.wrappers import KerasClassifier 
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
# Load the preprocessed dataset from a pickle file
df = pickle.load(open('pickles/data.pickle', 'rb'))

# Separate the features (X) and the target labels (y) from the dataset
X = df.drop(columns=['Target'])
y = df['Target']

# Split the data into training and testing sets (70/30 split), maintaining class balance with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=77, stratify=y)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

joblib.dump(scaler, "./joblibs/MLP_Scaler.joblib")

# Define a function to create a customizable Keras model (MLP) with hyperparameters for tuning
def create_model(hidden_layers=1, nodes_per_layer=32, activation='relu', learning_rate=0.001, optimizer='adam'):
    model = tf.keras.Sequential()
    # Add input layer based on the number of features in the dataset
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    
    # Add the specified number of hidden layers, each with the specified number of nodes and activation function
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation))
    
    # Add the output layer with 43 nodes (one for each class) and softmax activation for multiclass classification
    model.add(tf.keras.layers.Dense(43, activation='softmax'))

    # Choose optimizer (Adam or SGD) based on input hyperparameters
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Compile the model with Sparse Categorical Crossentropy loss and accuracy as the evaluation metric
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

# Wrap the Keras model in a Scikit-learn compatible interface (KerasClassifier) to use with GridSearchCV
model = KerasClassifier(
    model=create_model,
    verbose=1
)

# Define a grid of hyperparameters to search for optimal configurations using GridSearchCV
param_grid = {
    "epochs": [20],
    "batch_size": [32],
    "model__hidden_layers": [2],
    "model__nodes_per_layer": [32],
    "model__activation": ["relu"],
    "model__learning_rate": [0.001],
    "model__optimizer": ["sgd"],
}
# Use GridSearchCV to search for the best hyperparameter combination using 3-fold cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='accuracy', verbose=1)

# Train the model with the best combination of hyperparameters found by GridSearchCV
grid_result = grid.fit(X_train, y_train, class_weight=class_weights_dict)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Retrieve the best model after the grid search
best_model = grid_result.best_estimator_
test_loss, test_acc = best_model.model_.evaluate(X_test, y_test, verbose=2)
print(f'\nBest model test accuracy: {test_acc}')


y_pred = best_model.model_.predict(X_test)
# Convert predicted probabilities to class labels by selecting the index of the maximum probability
y_pred_classes = y_pred.argmax(axis=1)

# Generate a classification report showing precision, recall, F1-score, and support for each class
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_classes)}")

joblib.dump(best_model, 'joblibs/MLP_Model.joblib')
