import pandas as pd
import numpy as np
import scipy
import sklearn
import seaborn as sns   #visualiser les donnees
import matplotlib.pyplot as plt  #visualiser les données
# import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read original dataset
iris_df = pd.read_csv(r'C:\Users\MOUDIANGO-MOUDIANGO\OneDrive\Bureau\T\Iris.csv')
iris_df.sample(20)

X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
clf.fit(X_train, y_train.values.ravel())  # .values.ravel() converts DataFrame to 1D array

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

import joblib
# save the model to disk
joblib.dump(clf, "rf_model.sav")
