import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

import pickle

df = pd.read_csv("data.csv")
df.head()
df.isnull().sum()
print(df.shape)
print(df.describe)
df = df.drop(['Unnamed: 32'], axis=1)
df.head()
df = df.drop_duplicates()
df.head()
print(df.shape)
df.groupby('diagnosis').mean()
df = df.drop(['fractal_dimension_worst', 'smoothness_worst', 'symmetry_mean', 'smoothness_mean', 'symmetry_worst'], axis=1)
df.head()
df.groupby('diagnosis').mean()
df = df.drop(['fractal_dimension_mean', 'symmetry_se', 'fractal_dimension_se', 'id'], axis=1)
df.groupby('diagnosis').mean()
df = df.drop(['concavity_se', 'compactness_se', 'concave points_se', 'texture_se', 'compactness_mean', 'smoothness_se'], axis=1)
df.groupby('diagnosis').mean()
print(df.shape)
df['diagnosis'] = df['diagnosis'].replace({'B': 0, 'M': 1})
df.head()

df.plot.scatter('radius_mean', 'diagnosis')
df.plot.scatter('texture_mean', 'diagnosis')
df.plot.scatter('perimeter_mean', 'diagnosis')
df.plot.scatter('area_se', 'diagnosis')

x = df.drop(['diagnosis'], axis=1)
x.head()
y = df['diagnosis']
y.head()

models = []
seed = 4
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Random Forest Classification', RandomForestClassifier(n_estimators=100)))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# evaluate each model in turn
results = []
pre = []
recall_score = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    precision = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='precision')
    recall = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='recall')
    results.append(cv_results)
    pre.append(precision)
    recall_score.append(recall)
    names.append(name)
for i in range(6):
    print(names[i], ": ", results[i], " ", pre[i], " ", recall_score[i])

clf = RandomForestClassifier(n_estimators=300, random_state=0)
prediction = clf.fit(X_train, y_train)

pickle.dump(clf, open('random_forest_model.pkl', 'wb'))
