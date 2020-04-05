#importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

plt.scatter(X_train,y_train)
plt.title("Visualization of Data for finding Hypothesis class(Training Set)")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,color="red")
plt.title("The Best Fit line for given dataset(Testing set)")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()


