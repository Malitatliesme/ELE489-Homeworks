#Importing numpy and counter for necessary operations
import numpy as np
from collections import Counter

#Defining the KNN classifier
class KNN:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric  # "euclidean" or "manhattan"

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train)

    def _distance(self, x1, x2):
        #Computing for Euclidean
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        #Computing for Manhattan
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2))

    def predict(self, X_test):
        #Creating a a list to store predictions
        predictions = []
        for x in X_test:
            #Get distance from the test sample to the training sample
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            #Get indices of nearest neighbours
            k_neighbors_indices = np.argsort(distances)[:self.k]
            #Get labels of nearest neighbours
            k_neighbors_labels = [self.y_train[i] for i in k_neighbors_indices]
            #Determining most common label among neighbours
            most_common = Counter(k_neighbors_labels).most_common(1)
            #Appending most common class to predictions list
            predictions.append(most_common[0][0])
        return np.array(predictions)

    def accuracy(self, y_test, y_pred):
        return np.mean(y_test == y_pred)
