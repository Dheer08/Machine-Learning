import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")

x = data.iloc[:,1:4].values
y = data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
x[:,0] = le_gender.fit_transform(x[:,0])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3,metric="minkowski",p=2)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

acc = classifier.score(x_test,y_test)
print("acc : ",acc)


