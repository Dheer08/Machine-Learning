#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")
#print(dataset.head())

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
X_new_enc = pd.get_dummies(X[:,3])
X_new_enc = X_new_enc.to_numpy()
X = np.delete(X,3,1)
X = np.concatenate((X,X_new_enc),axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)
print("Acc : ",acc)
