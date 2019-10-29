import math

class KNeighborsClassifier:

    def __init__(self, k):
        self.k = k
    
    def distance(self,a,b):
        """Returns the Euclidean distance between a and b"""
        dim = len(a)
        sum = 0

        for d in range(dim):
            elem = (a[d]-b[d])**2
            sum = sum + elem
        return math.sqrt(sum)
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        
        for x in X_test:
            distance = []

            for i in range(len(self.X_train)):
                d = self.distance(x,self.X_train[i])
                distance.append((d,self.y_train[i]))

            distance.sort(key=lambda x:x[0])


            class_label = {}
            for i in range(self.k):
                ind = distance[i][1]
                if ind in class_label.keys():
                    class_label[ind]  = class_label[ind]+1
                else:
                    class_label[ind] = 1
            class_max = max(class_label,key = lambda k :class_label[k])
            predictions.append(class_max)


    def score(self, X_test, y_test):
        score = 0
        for i in range(len(X_test)):
            if X_test[i].all() == y_test[i].all():
                score = score + 1
        score = score / len(X_test)
        return score


