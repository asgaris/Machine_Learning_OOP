import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, folderPath):
        self.folderPath = folderPath
        self.read_data()
        self.clean_data()
        self.split_data()
    
    #Read data
    def read_data(self):
        csv_path = os.path.join(self.folderPath, 'bank.csv')
        self.data = pd.read_csv(csv_path)
    
    #Cleaning data
    def clean_data(self):
        # Convert categorical data to numerical data
        encoder = LabelEncoder()
        attributes = ['y', 'job', 'marital', 'education', 'loan']
        for att in attributes:
            self.data[att] = encoder.fit_transform(self.data[att])
        
        self.data.columns = [x.lower() for x in self.data.columns]

        #Features
        self.x = self.data[['age', 'job', 'marital', 'education', 'balance', 'loan']]
        #Target
        self.y = self.data[['y']].values.ravel()

        #Normalize data
        scaler = MinMaxScaler()
        scaler = scaler.fit(self.x)
        self.X = scaler.transform(self.x)

    #Split the data into a training and testing sets
    def split_data(self, test_size = 0.3):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size)

