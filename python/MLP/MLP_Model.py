import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scikeras.wrappers import KerasClassifier 
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


df = pickle.load(open('./pickles/data.pickle', 'rb'))

X = df.drop(columns=['Target'])
y = df['Target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77, stratify=y)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

def create_model(hidden_layers=1, nodes_per_layer=32, activation='relu', learning_rate=0.001, optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    

    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation))
    

    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

model = KerasClassifier(
    model=create_model,
    verbose=1
)

param_grid = {
    'epochs': [20],  
    'batch_size': [32],
    'model__hidden_layers': [2], 
    'model__nodes_per_layer': [32],  
    'model__activation': ['relu'],  
    'model__learning_rate': [0.001],  
    'model__optimizer': ['sgd'],  
}


grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)


grid_result = grid.fit(X_train, y_train, class_weight=class_weights_dict)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

best_model = grid_result.best_estimator_
test_loss, test_acc = best_model.model_.evaluate(X_test, y_test, verbose=2)
print(f'\nBest model test accuracy: {test_acc}')


y_pred = best_model.model_.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)


print(f"\nClassification Report:\n{classification_report(y_test, y_pred_classes)}")

with open('./pickles/MLP_Model.pickle', 'wb') as f:
    pickle.dump(best_model.model_, f)

