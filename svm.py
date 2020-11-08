
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('DM-Train.csv')
print(df.head())
print(df.columns)

x = df.iloc[:, 0:11]
print(x)

le = LabelEncoder()
ohe = OneHotEncoder()

x['Sex'] = le.fit_transform(x['Sex'])
x['Age'] = le.fit_transform(x['Age'])
x['Career'] = le.fit_transform(x['Career'])
x['Income'] = le.fit_transform(x['Income'])
x['Budget'] = le.fit_transform(x['Budget'])
x['Usage characteristics No1'] = le.fit_transform(x['Usage characteristics No1'])
x['Usage characteristics No2'] = le.fit_transform(x['Usage characteristics No2'])
x['Usage characteristics No3'] = le.fit_transform(x['Usage characteristics No3'])
x['Design characteristics No1'] = le.fit_transform(x['Design characteristics No1'])
x['Design characteristics No2'] = le.fit_transform(x['Design characteristics No2'])
x['Design characteristics No3'] = le.fit_transform(x['Design characteristics No3'])
print(x)

x = ohe.fit_transform(x).toarray()
print(x.shape)
print(x)

# f = open('test02.csv', 'w', encoding='utf8')
# for i in range(len(x)):
#     for j in range(77):
#         f.write(str(int(x[i][j])))
#         f.write(';')
#     f.write('\r')

df = pd.read_csv("01.csv")

x = df.iloc[:, 0:75]
y = df.iloc[:, -1]

from sklearn import svm

clf_svm = svm.SVC(class_weight='balanced')
clf_svm.fit(x,y)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_pred = cross_val_predict(clf_svm, x, y, cv=10)

scores = cross_val_score(clf_svm, x, y, cv=10)
print("cross-validated scores : ", scores)

avg_score = np.mean(scores)
print("\ncross-validated avg score : ", avg_score)

confusion_matrix = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print("\n", confusion_matrix)

print("\n", classification_report(y, y_pred))

import pickle
pkl_filename = "model_adidas_SVM.pkl"

with open(pkl_filename, 'wb') as file:
  pickle.dump(clf_svm, file)