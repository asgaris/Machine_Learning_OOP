from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer

class MachineLearningModel:
    def __init__(self, ML):
        #Grid search to find the best patameters
        param_grid = {
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'C': [0.01, 0.1, 1, 100, 1000],
            'max_iter': [100, 200, 300] 
            }
        self.search = GridSearchCV(estimator = ML, param_grid = param_grid, 
                            cv = 3, n_jobs = -1,  refit=True,
                            scoring=make_scorer(roc_auc_score),
                            verbose = 1)

    def fit_model(self, x_train, y_train):
        self.model = self.search.fit(x_train, y_train)
        print("tuned hpyerparameters :(best parameters) ", self.search.best_params_)
        print("accuracy :", self.search.best_score_)
    
    def predict(self, x_test):
        result = self.model.predict(x_test)
        return result
    