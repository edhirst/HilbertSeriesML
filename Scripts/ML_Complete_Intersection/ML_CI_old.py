'''Script to ML whether a Hilbert Series (HS) is a complete intersection (CI) or not, from inputs either:
    1) HS numerator coefficients:
        1a) where numertor is not palindromic
        1b) where numerator is palindromic
    2) HS Taylor expansion coefficients:
        2a) where coefficents are for Taylor expansions to order 20
        2b) where coefficents are for Taylor expansions to order 100
Each of these options determines 1 of the 4 investigations, these are demarkated in the script by a line of '#'s
...within each investigation there may be the option to use either: Neural Networks (NN), or Random Forests (RF), for the ML
...in addition the final cell of the script runs 5-fold cross-validation for whichever investigation was last run (i.e. however 'clf' was last defined)

To run:
    ~ Run the first cell to import the libraries
    ~ Choose an investigation from {1a,1b,2a,2b} to complete, & run the cells sequentially within each investigaions corresponding set of cells
        ...note to ensure the CI.db filepath is correct, and where there is an option to use NN or RF run only the cell corresponding to the ML structure you want to use
        ...also for investigation (1b) there is a second cell set that performs optimal hyperparameter search for the RF before ML
    ~ Run the script's final cell to perform a 5-fold cross-validation on the investigation last run
'''

#Import Libraries
import sqlite3
from ast import literal_eval
import pandas
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

#############################################################################################################
#%% (1a) #ML CI property from HS numerator coefficients using NNs (where these coefficients are not palindromic) 
#Import database
with sqlite3.connect("../../Data/CI.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT hs,ci FROM numerator_np", db)
# cast 'hilb'-column from a column of strings into a column of lists of ints
df['hs'] = df['hs'].transform(literal_eval)
hs=df['hs'].to_list();
ci=df['ci'].to_list();

#Separate data into train & test
hs_train, hs_test, ci_train, ci_test = train_test_split(df['hs'].to_list(), df['ci'].to_list(), test_size=0.2,shuffle=True)

#Create & train NN (aka MLP classifier)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
clf.fit(hs_train, ci_train)

#Test NN & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#############################################################################################################
#%% (1b) #ML CI property from HS numerator coefficients using NNs & RFs (where these coefficients are palindromic) 
#Import database
with sqlite3.connect("../Data/CI.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT hs,ci FROM numerator", db)
# cast 'hilb'-column from a column of strings into a column of lists of ints
df['hs'] = df['hs'].transform(literal_eval)
hs=df['hs'].to_list();
ci=df['ci'].to_list();

#Separate data into train & test
hs_train, hs_test, ci_train, ci_test = train_test_split(df['hs'].to_list(), df['ci'].to_list(), test_size=0.1,shuffle=True)

#%% #ML using NNs
#Create & train NN (aka MLP classifier)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
clf.fit(hs_train, ci_train)

#Test NN & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#%% #ML using RFs
#Create & train RF
clf = RandomForestClassifier(n_estimators=70, max_depth=70)
clf.fit(hs_train, ci_train)

#Test RF & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#############################################################################################################
#%% (1b).II #Perform a hyperparameter search to find optimal choices for the RF when learning from HS palindromic numerator coefficients
#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 150, num = 10)]
#Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
#Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(50, 150, num = 10)]
max_depth.append(None)
#Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
#Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
#Method of selecting samples for training each tree
bootstrap = [True, False]
#Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

#Use the random grid to search for best hyperparameters
#First create the base model to tune
rf = RandomForestClassifier()
#Random search of parameters, using cv fold cross validation, 
#Search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=0, random_state=None, n_jobs = None)
#Fit the random search model
rf_random.fit(hs, ci)
#Output the optimal hyperparameters
rf_random.best_params_

#%% #ML using RF with optimal hyperparameters
#Create & train RF (with optimal hyperparameters)
clf = RandomForestClassifier(n_estimators=105, max_depth=150,min_samples_split=2,min_samples_leaf=1,bootstrap=False)
clf.fit(hs_train, ci_train)

#Test RF & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#############################################################################################################
#%% (2a) #ML CI property from HS Taylor expansion coefficients using NNs & RFs (using coefficients to order 20)
#Import Dataset
with sqlite3.connect("../Data/CI.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT hs,ci FROM taylor_20", db)
# cast 'hilb'-column from a column of strings into a column of lists of ints
df['hs'] = df['hs'].transform(literal_eval)
# normalisation: divide entries by last entry and remove last entry
df['hs'] = df['hs'].transform(lambda h: [h[i] / h[-1] for i in range(0, len(h) - 1)])
hs=df['hs'].to_list();
ci=df['ci'].to_list();

#Separate data into train & test
hs_train, hs_test, ci_train, ci_test = train_test_split(df['hs'].to_list(), df['ci'].to_list(), test_size=0.2,shuffle=True)

#%% #ML using NNs
#Create & train NN (aka MLP classifier)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
clf.fit(hs_train, ci_train)

#Test NN & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#%% #ML using RFs
#Create & train RF
clf = RandomForestClassifier(n_estimators=100, max_depth=100)
clf.fit(hs_train, ci_train)

#Test RF & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#############################################################################################################
#%% (2b) #ML CI property from HS Taylor expansion coefficients using NNs & RFs (using coefficients to order 100)
#Import Dataset
with sqlite3.connect("../Data/CI.db") as db:
    c = db.cursor()
    df = pandas.read_sql_query("SELECT hs,ci FROM taylor_100", db)
# cast 'hilb'-column from a column of strings into a column of lists of ints
df['hs'] = df['hs'].transform(literal_eval)
# normalisation: divide entries by last entry and remove last entry
df['hs'] = df['hs'].transform(lambda h: [h[i] / h[-1] for i in range(0, len(h) - 1)])
hs=df['hs'].to_list();
ci=df['ci'].to_list();

#Separate data into train & test
hs_train, hs_test, ci_train, ci_test = train_test_split(df['hs'].to_list(), df['ci'].to_list(), test_size=0.2,shuffle=True)

#%% #ML using NNs
#Create & train NN (aka MLP classifier)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
clf.fit(hs_train, ci_train)

#Test NN & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#%% #ML using RFs
#Create & train RF
clf = RandomForestClassifier(n_estimators=100, max_depth=100)
clf.fit(hs_train, ci_train)

#Test RF & print learning metrics
ci_pred = clf.predict(hs_test)
print(f'MCC: {matthews_corrcoef(ci_test, ci_pred)}')
print(f'accuracy score: {accuracy_score(ci_test, ci_pred)}')

#############################################################################################################
#%% #Cross-validation on last defined clf (run after an investigation of choice to then perform cross-validation on that NN or RF structure) 
#Perform full 5-fold cross-validation training & testing (with 95% confidence interval)
scores = cross_val_score(clf, hs, ci, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
