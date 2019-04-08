# Pacotes
from os import path, chdir
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Garante que está no diretório raiz
abspath = path.abspath(__file__)
dname = path.dirname(abspath)
chdir("\\".join(dname.split("\\")[:-1]))

# Extracting data and formatting
train_data = pd.read_csv('./data/sonar.train-data', header=None)
X_train = train_data.iloc[:,0:-1].to_numpy(copy=True)
y_train = train_data.iloc[:,-1:].applymap(lambda s : 1 if s == "M" else 0).to_numpy(copy=True)

test_data = pd.read_csv('./data/sonar.test-data', header=None)
X_test = test_data.iloc[:,0:-1].to_numpy(copy=True)
y_train = test_data.iloc[:,-1:].applymap(lambda s : 1 if s == "M" else 0).to_numpy(copy=True)
