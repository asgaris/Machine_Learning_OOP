from Machine_Learnning_Model import MachineLearningModel
from Data_Preparation import DataLoader
from sklearn.linear_model import LogisticRegression

data = DataLoader(folderPath = "C:/Users/AsgariSa")
ML = MachineLearningModel(LogisticRegression())
ML.fit_model(data.x_train, data.y_train)
print(ML.predict(data.x_test))
