from annikaPotato import *

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the data
data = pd.read_csv('creditcard.csv')

# standardize the data
X = data.iloc[:, : -1]
y = data.iloc[:, -1]
XStandard = pd.DataFrame(StandardScaler().fit(X).transform(X), columns = X.columns[:])

# split the data set into training and testing ones
X_Train, X_Test, y_Train, y_Test = train_test_split(XStandard, y, test_size=0.3, 
                                                    random_state=RANDOM_SEED, stratify=y)

# ===============================
# gradient boosted decision tree
# ===============================
with open(NP_NUM_FILE_PATH, 'w') as fhand:
    fhand.write(f'=======\ngradient boosted decision tree:\n')

with open(NP_FRAUDULENT_CLASSES_PATH, 'w') as fhand:
    fhand.write(f'=======\ngradient boosted decision tree:\n')

print('GBDT')
for i in range(1, 10):
    print(i)
    gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=i,
                                      random_state=RANDOM_SEED)
    gbdt.fit(X_Train, y_Train)
    y_Pred = gbdt.predict(X_Test)
    aucScore = roc_auc_score(y_Test, gbdt.predict_proba(X_Test)[:, 1])

    writeToFile(NP_FRAUDULENT_CLASSES_PATH, 'a',
                getFraudulentClass(
                    f'max_depth = {i}\n',
                    np.where(y_Pred == 1)))

    writeToFile(NP_NUM_FILE_PATH, 'a',
                getStatistic(
                    f'max_depth = {i}\n',
                    aucScore,
                    y_Test,
                    y_Pred))

# ===============================
# logistic regression
# ===============================
with open(NP_NUM_FILE_PATH, 'a') as fhand:
    fhand.write(f'\n=======\nlogistic regression:\n')

with open(NP_FRAUDULENT_CLASSES_PATH, 'a') as fhand:
    fhand.write(f'\n=======\nlogistic regression:\n')

print('logistic regression')
for i in range(0, 100, 10):
    print(i)
    lr = LogisticRegression(C = i if i else i + 1, n_jobs=-1)
    lr.fit(X_Train, y_Train)
    y_Pred = lr.predict(X_Test)
    aucScore = roc_auc_score(y_Test, lr.predict_proba(X_Test)[:, 1])

    writeToFile(NP_FRAUDULENT_CLASSES_PATH, 'a',
                getFraudulentClass(
                    f'C = {i if i else i + 1} \n',
                    np.where(y_Pred == 1)))

    writeToFile(NP_NUM_FILE_PATH, 'a',
                getStatistic(
                    f'C = {i if i else i + 1} \n', 
                    aucScore,
                    y_Test,
                    y_Pred))                                                          

# ===============================
# random forest
# ===============================
with open(NP_NUM_FILE_PATH, 'a') as fhand:
    fhand.write(f'=======\nrandom forest:\n')

with open(NP_FRAUDULENT_CLASSES_PATH, 'a') as fhand:
    fhand.write(f'=======\nrandom forest:\n')

print('RF')
for i in range(1, 10):
    print(i)
    rf = RandomForestClassifier(n_estimators=100, max_depth=i,
                                      random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_Train, y_Train)
    y_Pred = rf.predict(X_Test)
    aucScore = roc_auc_score(y_Test, rf.predict_proba(X_Test)[:, 1])

    writeToFile(NP_FRAUDULENT_CLASSES_PATH, 'a',
                getFraudulentClass(
                    f'max_depth = {i}\n',
                    np.where(y_Pred == 1)))

    writeToFile(NP_NUM_FILE_PATH, 'a',
                getStatistic(
                    f'max_depth = {i}\n',
                    aucScore,
                    y_Test,
                    y_Pred))