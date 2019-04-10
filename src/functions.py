import numpy as np
import pandas as pd

def split_data_labels(pandas_table):
  X = pandas_table.iloc[:,0:-1].to_numpy(copy=True)
  y = pandas_table.iloc[:,-1:].applymap(lambda s : 1 if s == "M" else -1).to_numpy(copy=True)
  return (X,y)

class Perceptron:
  def __init__(self, n_inputs, learning_rate):
    self.learning_rate = learning_rate
    self.weights = np.transpose(np.random.rand(n_inputs))
    self.biases = np.random.rand()
    self.best_weights = self.weights
    self.best_biases = self.biases
    self.best_error = 1

  def forward_pass(self, X):
    return 1 if np.dot(self.weights, X.T) + self.biases > 0 else -1

  def calculate_delta(self, X, y, d):
    w = self.learning_rate*X*(y-d)
    b = self.learning_rate*1*(y-d)
    return w, b

  def correct(self, delta_w, delta_b):
    self.weights += delta_w
    self.biases += delta_b

  def get_accuracy(self, X_test, y_test):
    correct = 0
    for X, y in zip(X_test,y_test):
      d = self.predict(X)
      correct += 1 if d == y else 0
    acc = correct/X_test.shape[0]
    err = 1-acc
    if self.best_error > err:
      self.best_error = err
      print(self.best_error)
      self.best_weights = np.copy(self.weights)
      self.best_biases = np.copy(self.biases)
    return acc, err

  def train_epoch(self, X_train, y_train):
    delta_w_sum = 0
    delta_b_sum = 0
    for X, y  in zip(X_train,y_train):
      d = self.forward_pass(X)
      delta_w, delta_b = self.calculate_delta(X, y, d)
      delta_w_sum += delta_w
      delta_b_sum += delta_b
    self.correct(delta_w_sum, delta_b_sum)

  def fit(self, X_train, y_train, epochs=1, X_validation=None, y_validation=None):
    # Train for N epochs
    results = []
    for i in range(epochs):
      self.train_epoch(X_train, y_train)
      if X_validation.any() and y_validation.any():
        results.append(self.get_accuracy(X_validation, y_validation))
    return results

  def predict(self, X):
    return self.forward_pass(X)

  def best_predict(self, X):
    return 1 if np.dot(self.best_weights, X.T) + self.best_biases > 0 else -1

  def get_best_accuracy(self, X_test, y_test):
    correct = 0
    for X, y in zip(X_test,y_test):
      d = self.best_predict(X)
      correct += 1 if d == y else 0
    acc = correct/X_test.shape[0]
    err = 1-acc
    return acc, err

# class MultilayerPerceptron:
#   def __init__(self, structure, learning_rate):
#     self.structure = structure
#     self.learning_rate = learning_rate
#     self.weights = []
#     self.biases = []
#     for i in range(1,len(structure)):
#       # Cada linha é 1 perceptron, cada coluna é uma entrada
#       weight_matrix = np.random.rand(structure[i],structure[i-1])
#       self.weights.append(weight_matrix)
#       self.biases.append(np.random.rand())
#     #print(self.weights)

#   def activation(self, z):
#     return z

#   def activation_deriv(self, z):
#     return 1

#   def forward_pass(self, X):
#     z = [] # Weighted sum
#     h = [] # Activation function of z
#     h.append(X.T)
#     for i, (w, b) in enumerate((self.weights, self.biases)):
#       z.append(np.dot(w, h[i]) + b)
#       h.append(self.activation(z[i+1]))
#     return h, z

#   # -gradient(z)*error
#   def calculate_out_layer_delta(self, y, h_out, z_out):
#     return -(y-h_out) * self.activation_deriv(z_out)

#   # gradient(z)*w_{l}^{T}.delta_{l+1}
#   def calculate_hidden_delta(self, delta_plus_1, w_l, z_l):
#     return np.dot(w_l.T, delta_plus_1) * self.activation_deriv(z_l)

#   def backpropagate(self, y_train, h, z):
#     delta = {}
#     for l, s in reversed(list(enumerate(self.structure))):
#       if l == len(self.structure):
#         delta[l] = self.calculate_out_layer_delta(y_train, h[l], z[l])
#         avg_cost += np.linalg.norm((y_train-h[l]))
#       else:
#         if l > 1:
#           delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
#         delta_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:, np.newaxis]))
#         delta_b[l] += delta[l+1]

#   def train_epoch(self, X_train, y_train, batch=1):
#     cont = 0
#     while cont != len(X_train):
#       # Forward pass batch times
#       results = []
#       for i in range(batch):
#         out = self.forward_pass(X_train[cont])
#         results.append((out, y_train[cont]))
#         cont += 1
#       # Backpropagate
#       for result in results:

#     #return accuracy
#     pass

#   def fit(self, X_train, y_train, epochs=10, batch=1):
#     pass
#     # Train for N epochs
#     #return self.weights, accuracy

#   def predict(self, X):
#     return 0