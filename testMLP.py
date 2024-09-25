import joblib
import pickle
import tensorflow as tf
from sklearn.metrics import classification_report

# Define the create_model function as before
def create_model(hidden_layers=1, nodes_per_layer=32, activation='relu', learning_rate=0.001, optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))  # Adjust as needed based on feature shape
    
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation))
    
    model.add(tf.keras.layers.Dense(43, activation='softmax'))  # Adjust output size as per your model
    
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

# Load the model and scaler
model = joblib.load("./joblibs/MLP_Model.joblib")
scaler = joblib.load("./joblibs/MLP_Scaler.joblib")

# Load the validation data
df = pickle.load(open('pickles/validationData.pickle', 'rb'))

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Scale the features using the loaded scaler
X_scaled = scaler.transform(X)

print("prediciting")
# Make predictions
prediction = model.predict(X_scaled)
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
