#ML - tree (CERT)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

col_names = []

with open('row_names.txt', 'r') as f:
	line = f.readline().strip()
	while line:
		col_names.append(line)
		line = f.readline().strip()

income_set = pd.read_csv('census-income.csv', names=col_names)

for x in col_names:
	not_in_uni = income_set[x] == ' Not in universe'
	income_set.loc[not_in_uni, x] = np.nan

n = income_set.select_dtypes(include='float64')
X = n.drop('instance weight', axis=1)
y = income_set['income class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45, stratify=y)

income_set_test = pd.read_csv('census-income-test.csv', names = col_names)
pred_set = income_set_test.select_dtypes(include='int64')


dt = DecisionTreeClassifier(max_depth=18, random_state =45)
dt.fit(X_train, y_train)

print (dt.score(X_test, y_test))

prediction = dt.predict(pred_set)
dictionary = {' - 50000.': 0, ' 50000+.': 1}
prediction = pd.DataFrame(prediction)#, names = ['index', 'income class'])

#print (prediction.info())

prediction[0] = prediction[0].map(dictionary)
prediction.to_csv('prediction.csv')