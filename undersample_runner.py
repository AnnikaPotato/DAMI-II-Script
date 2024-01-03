# import random
from annikaPotato import *
from processed_ml import Runner
from sklearn.utils import resample, shuffle

FRAUD_TRANSACTION = 492

# load the data
data = pd.read_csv('creditcard.csv')

genuineTran = data[data['Class'] == 0]
fraudTran = data[data['Class'] == 1]

ori_X = data.iloc[:, : -1]
ori_y = data.iloc[:, -1]
standarizedAll = pd.DataFrame(StandardScaler().fit(ori_X).transform(ori_X), columns = ori_X.columns[:])
runner = Runner()

for i in [1, 5, 10, 50, 100]:
    size = i * FRAUD_TRANSACTION
    selectedGenuine = resample(genuineTran, 
                               replace=True, 
                               n_samples= size)

    shuffledArray = shuffle(pd.concat([fraudTran, selectedGenuine],axis=0))
    X = shuffledArray.iloc[:, : -1]
    y = shuffledArray.iloc[:, -1]
    XStandard = pd.DataFrame(StandardScaler().fit(X).transform(X), columns = X.columns[:])
    X_Train, X_Test, y_Train, y_Test = train_test_split(XStandard, y, test_size=0.3, 
                                                        random_state=RANDOM_SEED, stratify=y)

    runner.set(X_Train, X_Test, y_Train, y_Test, f'us_{i}')
    runner.run()