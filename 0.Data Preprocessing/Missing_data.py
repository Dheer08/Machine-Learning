import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data.csv")
#print(data.head())

X = data.iloc[:,:-1].values
Y  =data.iloc[:,3].values


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan,strategy='mean')
imp = imp.fit(X[:,1:3])
X[:,1:3] = imp.transform(X[:,1:3])

print(X)