# Pacotes
from os import path, chdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from functions import *

# Garante que está no diretório raiz
abspath = path.abspath(__file__)
dname = path.dirname(abspath)
chdir("\\".join(dname.split("\\")[:-1]))

# Extracting data and formatting (Labels are 1 for M and 0 for R)
train_data = pd.read_csv('./data/sonar.train-data', header=None)
test_data = pd.read_csv('./data/sonar.test-data', header=None)
X_train, y_train = split_data_labels(train_data)
X_test, y_test = split_data_labels(test_data)

# Configure
per = Perceptron(X_train.shape[1], 0.00001) # Inicia Perceptron com 60 entradas
#per = MultilayerPerceptron((X_train.shape[1],1), 0.1)
# Train
results = per.fit(X_train, y_train, epochs=10, X_validation=X_test, y_validation=y_test)
acc, err = zip(*results)
print(min(*err))
print(per.get_best_accuracy(X_test,y_test))
plt.plot(err)
plt.ylim(0,1)
plt.xlim(left=0)
plt.show()
