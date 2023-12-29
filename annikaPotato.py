RANDOM_SEED = 938773
CONSUFION_MATRIX = (('TP', 'FN'), ('FP', 'TN'))
NP_NUM_FILE_PATH = 'resultStatistic_NoPre.txt'
NP_FRAUDULENT_CLASSES_PATH = 'fraudulent_NoPre.txt'

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

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