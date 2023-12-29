from annikaPotato import *

class Runner:
    def __init__(self) -> None:
        self.X_Train = None
        self.X_Test = None
        self.y_Train = None
        self.y_Test = None
        pass

    def set(self, X_Train, X_Test, y_Train, y_Test, filesuffix) -> None:
        self.X_Train = X_Train
        self.X_Test = X_Test
        self.y_Train = y_Train
        self.y_Test = y_Test
        P_FRAUDULENT_CLASSES_PATH = f'resultStatistic_P_{filesuffix}'
        P_NUM_FILE_PATH = f'fraudulent_P_{filesuffix}'
        return
    
    def run(self) -> None:
        self.runGBDT()
        self.runLR()
        self.runRF()
        return
    
    def runGBDT(self) -> None:
        with open(P_NUM_FILE_PATH, 'w') as fhand:
            fhand.write(f'=======\ngradient boosted decision tree:\n')

        with open(P_FRAUDULENT_CLASSES_PATH, 'w') as fhand:
            fhand.write(f'=======\ngradient boosted decision tree:\n')
        
        for i in range(MIN_DEPTH, MAX_DEPTH + INTERVAL_DEPTH, INTERVAL_DEPTH):
            print(i)
            gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=i,
                                            random_state=RANDOM_SEED)
            gbdt.fit(self.X_Train, self.y_Train)
            y_Pred = gbdt.predict(self.X_Test)
            aucScore = roc_auc_score(self.y_Test, gbdt.predict_proba(self.X_Test)[:, 1])

            writeToFile(P_FRAUDULENT_CLASSES_PATH, 'a',
                        getFraudulentClass(
                            f'max_depth = {i}\n',
                            np.where(y_Pred == 1)))

            writeToFile(P_NUM_FILE_PATH, 'a',
                        getStatistic(
                            f'max_depth = {i}\n',
                            aucScore,
                            self.y_Test,
                            y_Pred))

    def runLR(self) -> None:
        with open(P_NUM_FILE_PATH, 'a') as fhand:
            fhand.write(f'\n=======\nlogistic regression:\n')

        with open(P_FRAUDULENT_CLASSES_PATH, 'a') as fhand:
            fhand.write(f'\n=======\nlogistic regression:\n')

        for i in range(0, C_EXP):
            c = 5 * (0.1 ** i)
            print(c)
            lr = LogisticRegression(C = c, n_jobs=-1, 
                                    random_state=RANDOM_SEED)
            lr.fit(self.X_Train, self.y_Train)
            y_Pred = lr.predict(self.X_Test)
            aucScore = roc_auc_score(self.y_Test, lr.predict_proba(self.X_Test)[:, 1])

            writeToFile(NP_FRAUDULENT_CLASSES_PATH, 'a',
                        getFraudulentClass(
                            f'C = {c} \n',
                            np.where(y_Pred == 1)))

            writeToFile(NP_NUM_FILE_PATH, 'a',
                        getStatistic(
                            f'C = {c} \n', 
                            aucScore,
                            self.y_Test,
                            y_Pred))  
            
    def runRF(self):
        with open(P_NUM_FILE_PATH, 'a') as fhand:
            fhand.write(f'=======\nrandom forest:\n')

        with open(P_FRAUDULENT_CLASSES_PATH, 'a') as fhand:
            fhand.write(f'=======\nrandom forest:\n')

        print('RF')
        for i in range(MIN_DEPTH, MAX_DEPTH + INTERVAL_DEPTH, INTERVAL_DEPTH):
            print(i)
            rf = RandomForestClassifier(n_estimators=100, max_depth=i,
                                            random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(self.X_Train, self.y_Train)
            y_Pred = rf.predict(self.X_Test)
            aucScore = roc_auc_score(self.y_Test, rf.predict_proba(self.X_Test)[:, 1])

            writeToFile(NP_FRAUDULENT_CLASSES_PATH, 'a',
                        getFraudulentClass(
                            f'max_depth = {i}\n',
                            np.where(y_Pred == 1)))

            writeToFile(NP_NUM_FILE_PATH, 'a',
                        getStatistic(
                            f'max_depth = {i}\n',
                            aucScore,
                            self.y_Test,
                            y_Pred))