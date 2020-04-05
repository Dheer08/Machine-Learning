import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data.csv")
#print(data.head())

X = data.iloc[:,:-1].values
Y  =data.iloc[:,3].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
#Dummy Encoding
#onehotencoder = OneHotEncoder(categorical_features=[3])

X_dummy = pd.get_dummies(X[:,0])
X_dummy = X_dummy.to_numpy()
X = np.delete(X,0,axis=1)
X = np.concatenate((X,X_dummy),axis=1)

labelencoder_y = LabelEncoder()
Y = labelencoder.fit_transform(Y)
#print(Y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
print(x_train)