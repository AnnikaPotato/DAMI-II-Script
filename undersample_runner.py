# import random
from annikaPotato import *
from processed_ml import Runner
from sklearn.utils import resample, shuffle

FRAUD_TRANSACTION = 492

# load the data
data = pd.read_csv('creditcard.csv')

genuineTran = data[data['Class'] == 0]
fraudTran = data[data['Class'] == 1]
#fileSuffix = random.randint(1000, 9999)

for i in range(0, 3):
    size = 3 * (3 ** i) * FRAUD_TRANSACTION
    selectedGenuine = resample(genuineTran, 
                               replace=True, 
                               n_samples= size)

    shuffledArray = shuffle(pd.concat([fraudTran, selectedGenuine],axis=0))
    X = shuffledArray.iloc[:, : -1]
    y = shuffledArray.iloc[:, -1]
    XStandard = pd.DataFrame(StandardScaler().fit(X).transform(X), columns = X.columns[:])
    X_Train, X_Test, y_Train, y_Test = train_test_split(XStandard, y, test_size=0.3, 
                                                        random_state=RANDOM_SEED, stratify=y)

    runner = Runner()
    runner.set(X_Train, X_Test, y_Train, y_Test, f'us_{size}')
    runner.run()