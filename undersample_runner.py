import random
from annikaPotato import *
from processed_ml import Runner

# load the data
data = pd.read_csv('creditcard.csv')

# standardize the data
X = data.iloc[:, : -1]
y = data.iloc[:, -1]
XStandard = pd.DataFrame(StandardScaler().fit(X).transform(X), columns = X.columns[:])

X_Train, X_Test, y_Train, y_Test = train_test_split(XStandard, y, test_size=0.3, 
                                                    random_state=RANDOM_SEED, stratify=y)

runner = Runner()
runner.set(X_Train, X_Test, y_Train, y_Test, f'us_{random.randint(1000, 9999)}')
runner.run()

