import numpy as np
import pandas as pd

class Perceptron:
  def __init__(self, n_inputs, learning_rate):
    self.learning_rate = learning_rate
    self.weights = np.random.rand(n_inputs)
    self.biases = np.random.rand()
    self.best_weights = np.copy(self.weights)
    self.best_biases = np.copy(self.biases)
    self.best_error = 1

  @staticmethod
  def split_data_labels(pandas_table):
    X = pandas_table.iloc[:,0:-1].to_numpy(copy=True)
    y = pandas_table.iloc[:,-1:].applymap(lambda s : 1 if s == "M" else -1).to_numpy(copy=True)
    return (X,y)

  def forward_pass(self, X):
    return 1 if np.dot(self.weights, X) + self.biases > 0 else -1

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
#     self.best_weights = self.weights[:]
#     self.best_biases = self.biases[:]
#     self.best_error = 1

#   @staticmethod
#   def split_data_labels(pandas_table):
#     X = pandas_table.iloc[:,0:-1].to_numpy(copy=True)
#     y = pandas_table.iloc[:,-1:].applymap(lambda s : np.array([1,0])[:,np.newaxis] if s == "M" else np.array([0,1])[:,np.newaxis]).to_numpy(copy=True)
#     return (X,y)


#   def activation(self, z):
#     return 1/(1+np.exp(z))

#   def activation_deriv(self, z):
#     return self.activation(z) * (1-self.activation(z))

#   def forward_pass(self, X):
#     z = {} # Weighted sum
#     h = {0:X.T} # Activation function of z
#     for l in range(0, len(self.weights)):
#       z[l+1] = np.dot(self.weights[l], h[l]) + self.biases[l]
#       h[l+1] = self.activation(z[l+1])
#     return h, z

#   # -gradient(z)*error
#   def calculate_out_layer_delta(self, y, h_out, z_out):
#     return -(y-h_out) * self.activation_deriv(z_out)

#   # gradient(z)*w_{l}^{T}.delta_{l+1}
#   def calculate_hidden_delta(self, delta_plus_1, w_l, z_l):
#     return np.dot(w_l.T, delta_plus_1) * self.activation_deriv(z_l)

#   # Mudar para tudo começar do fucking 0
#   def calculate_backpropagation(self, y_train, h, z):
#     delta   = {}
#     delta_W = {}
#     delta_b = {}
#     for l in range(len(self.structure)-1, -1, -1):
#       if l == len(self.structure)-1:
#         delta[l] = self.calculate_out_layer_delta(y_train, h[l], z[l])
#       else:
#         if l > 0:
#           delta[l] = self.calculate_hidden_delta(delta[l+1], self.weights[l], z[l])
#         delta_W[l] = np.dot(delta[l+1][:,np.newaxis], h[l][np.newaxis,:])
#         delta_b[l] = delta[l+1]
#     return delta_W, delta_b

#   def backpropagate(self, batch, delta_W, delta_b):
#     for l in range(len(self.structure) - 2, -1, -1):
#       self.weights[l] += -self.learning_rate * delta_W[l]
#       #print('antes:',self.biases)
#       #print('backpropagate:',-self.learning_rate * delta_b[l])
#       self.biases[l]  += -self.learning_rate * delta_b[l]
#       #print('depois:',self.biases)

#   def get_accuracy(self, X_test, y_test):
#     correct = 0
#     for X, y in zip(X_test,y_test):
#       d = self.predict(X)
#       #print(d,y,abs(d - y) < 1)
#       correct += 1 if abs(d - y) < 1 else 0
#     acc = correct/X_test.shape[0]
#     err = 1-acc
#     if self.best_error > err:
#       self.best_error = err
#       print(self.best_error)
#       self.best_weights = np.copy(self.weights)
#       self.best_biases = np.copy(self.biases)
#     return acc, err

#   def train_epoch(self, X_train, y_train, batch):
#     cont = 0
#     while cont < len(X_train):
#       delta_W = {}
#       delta_b = {}
#       # Forward pass batch times
#       for i in range(batch):
#         h, z = self.forward_pass(X_train[cont])
#         ws, bs = self.calculate_backpropagation(y_train[cont], h, z)
#         #print(list(zip(ws.values(),bs.values())))
#         for l, (w, b) in enumerate(zip(ws.values(),bs.values())):
#           #print('train epoch',l,w,b)
#           try:
#             delta_W[l] += w
#             delta_b[l] += b
#           except:
#             delta_W[l] = w
#             delta_b[l] = b

#         cont += 1
#         if cont >= len(X_train):
#           break
#       # Apply backpropagation
#       self.backpropagate(batch, delta_W, delta_b)

#   def fit(self, X_train, y_train, epochs=10, batch=-1, X_validation=None, y_validation=None):
#     # Train for N epochs
#     results = []
#     for i in range(epochs):
#       self.train_epoch(X_train, y_train, batch if batch > 0 else X_train.shape[0])
#       if X_validation.any() and y_validation.any():
#         results.append(self.get_accuracy(X_validation, y_validation))
#         #print(results)
#     return results

#   def predict(self, X):
#     h, z = self.forward_pass(X)
#     y = np.argmax(h[len(h)-1])
#     return y

#   def best_predict(self, X):
#     h = {0:X.T}
#     for l in range(0, len(self.weights)):
#       h[l+1] = self.activation(np.dot(self.best_weights[l], h[l]) + self.best_biases[l])
#     return h[len(h)-1]

#   def get_best_accuracy(self, X_test, y_test):
#     correct = 0
#     for X, y in zip(X_test,y_test):
#       d = self.best_predict(X)
#       correct += 1 if abs(d - y) < 1 else 0
#     acc = correct/X_test.shape[0]
#     err = 1-acc
#     return acc, err
