import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

from sklearn.preproocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform()
y = sc_y.fit_transform()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_test)
