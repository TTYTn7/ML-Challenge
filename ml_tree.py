#ML - tree (CERT)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

col_names = []
# Adding names for the columns of the input data
'''
Ieva - I remember you had a better way of doing that in about 1 line, right?
Can you do so here?
'''
with open('row_names.txt', 'r') as f:
	line = f.readline().strip()
	while line:
		col_names.append(line)
		line = f.readline().strip()

income_set = pd.read_csv('census-income.csv', names=col_names)
#print (income_set.info())

# Replacing the default placeholder value ('Not in universe') with NaN
#for x in col_names:
#	not_in_uni = income_set[x] == ' Not in universe'
#	income_set.loc[not_in_uni, x] = np.nan


# Selecting all of the numerical fields as predictors, 'aside of instance weight'
# TO DO:
#		Find a way to add categorical (string) data as predictors
#X = income_set.select_dtypes(include='int64')
X = pd.get_dummies(income_set[['age', 'detailed industry recode', 
				'detailed occupation recode', 'education', 'wage per hour', 
				'capital gains', 'capital losses','dividends from stocks', 
				'num persons worked for employer', 'own business or self employed', 
				'veterans benefits', 'weeks worked in year', 'year']])
#print (X.info())
y = income_set['income class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42, stratify=y)

# Adding the test set, again only the numeric values
pred_set = pd.read_csv('census-income-test.csv', names=col_names)#.select_dtypes(include='int64')
pred_set_2 = pd.get_dummies(pred_set[['age', 'detailed industry recode', 
				'detailed occupation recode', 'education', 'wage per hour', 
				'capital gains', 'capital losses','dividends from stocks', 
				'num persons worked for employer', 'own business or self employed', 
				'veterans benefits', 'weeks worked in year', 'year']])
#print (pred_set_2.info())


dt = DecisionTreeClassifier(max_depth=36, random_state =42)
dt.fit(X_train, y_train)
print (dt.score(X_test, y_test))


prediction = pd.DataFrame(dt.predict(pred_set_2))

#print (prediction.info())

# Replacing the default vals with the ones required by the solution format
dictionary = {' - 50000.': 0, ' 50000+.': 1}
prediction[0] = prediction[0].map(dictionary)

prediction.to_csv('prediction.csv', header=['income class'])
# TO DO:
#		Add the 'index' and 'income class' col names to the prediction.csv upon saving.
#		Currently has to be done manually