import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam, SGD

def create_MLP(hidden_layers=1, nodes_per_layer=32, activation='relu', learning_rate=0.001, optimizer='adam'):
  model = Sequential()

  # Define the input layer with the appropriate shape (3072, which is 32*32*3 flattened)
  model.add(Input(shape=(3072,)))

  # Input layer
  model.add(Dense(nodes_per_layer, activation=activation))

  # Hidden layers
  for _ in range(hidden_layers):
      model.add(Dense(nodes_per_layer, activation=activation))

  # Output layer: 43 nodes (one for each class), softmax activation for probability distribution
  model.add(Dense(5, activation='softmax'))

  # Compile the model
  if optimizer == 'adam':
      optimizer = Adam(learning_rate=learning_rate)
  elif optimizer == 'sgd':
      optimizer = SGD(learning_rate=learning_rate)

  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model
