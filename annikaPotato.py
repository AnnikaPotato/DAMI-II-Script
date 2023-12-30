import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

RANDOM_SEED = 938773

MIN_DEPTH = 1
MAX_DEPTH = 10
INTERVAL_DEPTH = 1

C_EXP = 5

CONSUFION_MATRIX = (('TP', 'FN'), ('FP', 'TN'))
NP_NUM_FILE_PATH = 'resultStatistic_NoPre.txt'
NP_FRAUDULENT_CLASSES_PATH = 'fraudulent_NoPre.txt'

def writeToFile(path: str, mode: str, content: str) -> None:
    with open(path, mode) as fhand:
        fhand.write(content)
    return

def getFraudulentClass(header: str, contents) -> str:
    classIndice = ','.join(map(str, contents[0]))
    return ''.join([header, classIndice, '\n\n'])

def getStatistic(header: str, aucScore: float, y_Test, y_Pred) -> str:
    toWrite = [header]

    # save the confusion matrix
    for r, row in enumerate(confusion_matrix(y_Test, y_Pred)):
        for c, col in enumerate(row):
            toWrite.append(f'\t{CONSUFION_MATRIX[r][c]}: {col}\t')
    toWrite.append('\n')

    toWrite.append(f'\tAccuracy: {accuracy_score(y_Test, y_Pred)}\n')
    toWrite.append(f'\tPrecision: {precision_score(y_Test, y_Pred)}\n')
    toWrite.append(f'\tRecall: {recall_score(y_Test, y_Pred)}\n')
    toWrite.append(f'\tF1: {f1_score(y_Test, y_Pred)}\n')
    toWrite.append(f'\tAUC:\t{aucScore}\n')
    toWrite.append('\n')

    return ''.join(toWrite)