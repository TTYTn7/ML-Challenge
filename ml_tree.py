#ML - tree (CERT)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

training_set = pd.read_csv('census-income.csv', names=col_names)
#print (income_set.info())

# Replacing the default placeholder value ('Not in universe') with NaN
#for x in col_names:
#	not_in_uni = income_set[x] == ' Not in universe'
#	income_set.loc[not_in_uni, x] = np.nan


# Selecting all of the numerical fields as predictors, 'aside of instance weight'
# TO DO:
#		Find a way to add categorical (string) data as predictors
#X = income_set.select_dtypes(include='int64')
X = pd.get_dummies(training_set[['age', 'detailed industry recode', 
				'detailed occupation recode', 'education', 'wage per hour', 
				'capital gains', 'dividends from stocks', 'major occupation code',
				'major industry code', 'sex', 'class of worker',
				'num persons worked for employer', 'weeks worked in year']])
#print (X.info())
y = training_set['income class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 2, stratify=y)
#X_test = pd.get_dummies(test_set[['age', 'detailed industry recode', 
				#'detailed occupation recode', 'education', 'wage per hour', 
				#'capital gains', 'capital losses','dividends from stocks', 
				#'num persons worked for employer', 'own business or self employed', 
				#'veterans benefits', 'weeks worked in year']])
#y_test = test_set['income class'].values

# Adding the test set, again only the numeric values
test_set = pd.read_csv('census-income-test.csv', names=col_names)#.select_dtypes(include='int64')
test_set_2 = pd.get_dummies(test_set[['age', 'detailed industry recode', 
				'detailed occupation recode', 'education', 'wage per hour', 
				'capital gains', 'dividends from stocks', 'major occupation code',
				'major industry code', 'sex', 'class of worker',
				'num persons worked for employer', 'weeks worked in year']])
#print (pred_set_2.info())


#Single tree
# dt = DecisionTreeClassifier(max_depth=42, random_state =2)
# dt.fit(X_train, y_train)
# print ('Tree score:')
# print (dt.score(X_test, y_test))

#Forest
forest = RandomForestClassifier(n_estimators=200, bootstrap=True, max_features='sqrt')
forest.fit(X_train, y_train)
print ('Forest score:')
print (forest.score(X_test, y_test))

# prediction = pd.DataFrame(dt.predict(test_set_2))
prediction_forest = pd.DataFrame(forest.predict(test_set_2))

# fi = pd.DataFrame({'feature': list(training_set.columns), 'importance': dt.feature_importances_}).\
#                     sort_values('importance', ascending = False)

# fi2 = pd.DataFrame({'feature': list(training_set.columns), 'importance': forest.feature_importances_}).\
# 					sort_values('importance', ascending = False)

# print ('Feature importance tree:')
# print (fi.head())

# print ('Feature importance forest:')
# print (fi2.head())
#print (prediction.info())

# Replacing the default vals with the ones required by the solution format
dictionary = {' - 50000.': 0, ' 50000+.': 1}
prediction_forest[0] = prediction_forest[0].map(dictionary)

prediction_forest.to_csv('prediction.csv', header=['income class'])
# TO DO:
#		Add the 'index' and 'income class' col names to the prediction.csv upon saving.
#		Currently has to be done manually