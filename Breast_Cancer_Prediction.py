import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class'] = breast_cancer.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify = Y, random_state = 1)

classifier = LogisticRegression()

classifier.fit(X_train, Y_train)

prediction_test = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, prediction_test)

input_data = list(map(str,input("enter: ").split(",")))
data_array = np.asarray(input_data)
data_reshaped = data_array.reshape(1,-1)

prediction = classifier.predict(data_reshaped)
if prediction[0] == 0 :
    print("Malignant")
else:
    print("Benign")