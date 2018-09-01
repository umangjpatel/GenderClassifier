#Dataset is in the GenderClassifier directory

#Import the numpy package for efficiently working with list data from the dataset
import numpy as np
#Import the pandas package for reading the dataset csv file
import pandas as pd
#Import the logistic regression ML model
from sklearn.linear_model import LogisticRegression
#Import the accuracy_score package for finding the accuracy model
from sklearn.metrics import accuracy_score

#Storing the dataset after reading the csv file
dataset = np.array(pd.read_csv('../dataset.csv'))

#Lists for storing the dataset
#Features dataset in order of height, weight and shoe size
features = []
#Labels (Male/Female) for above features dataset
labels = []

#Fetching individual datapoints from the dataset and storing in our training data
for data in dataset:
    foot_size = float(data[0])
    height = int(data[1])
    gender = str(data[2])
    features.append([foot_size, height])
    labels.append(gender)

#Getting classifier of the Logistic regression ML model
clf = LogisticRegression()

#Training the model with the training data
clf.fit(features, labels)

#Getting inputs from users
#Input height
user_height = int(input("Enter the height (in cms.) : "))

#Input foot size
user_foot_size = int(input("Enter the foot size (in inches) : "))

#Combining user inputs as list of lists
inputs = [[user_foot_size, user_height]]

#Print the prediction result
print(clf.predict(inputs))

#Print the accuracy score of the ML model
print("Accuracy : %.2f percent" % (100 * accuracy_score(['Male', 'Male', 'Female'], ['Female', 'Female', 'Female'])))
