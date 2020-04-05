import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Social_Network_Ads.csv")

x = dataset.iloc[:,1:4].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
x[:,0] = le_gender.fit_transform(x[:,0])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn import svm
classifier = svm.SVC()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

acc = classifier.score(x_test,y_test)
print("acc : ",acc)
