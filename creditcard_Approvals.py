# In this project, I build an automatic credit card approval predictor using machine learning techniques to predict 
# if a person's application for a credit card would get approved or not given some information about that person.

#Using Credit Card Approval Dataset from UCI Machine Learning Repository.
#http://archive.ics.uci.edu/ml/datasets/Credit+Approval


#Start by loading and viewing the dataset.
#The columns are Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, 
#PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and ApprovalStatus.

#The dataset has 690 instances.
#The dataset contains both numeric and non-numeric data.
#column 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) 
#and all the other columns contain non-numeric values (of type object).

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Importing dataset using pandas
dataset = pd.read_csv(r'C:\Users\User\Desktop\Github_project\credit\cc.csv', header=None)

# Manipulating the data

#The dataset has missing values. The missing values in the dataset are labeled with '?'.
#Replace ? with NaN
dataset = dataset.replace('?',np.nan)

#impute the missing values with mean imputation in the numeric columns.
dataset.fillna(dataset.mean(), inplace = True)

#impute the missing values with the most frequent values as present in the non-numeric columns
for col in dataset:
    #check if cloumn is object type?
    if dataset[col].dtype == 'object':
        #impute the missing values with the most frequent values
        dataset = dataset.fillna(dataset[col].value_counts().index[0])

#Convert the non-numeric data into numeric data using a technique called label encoding.
le = preprocessing.LabelEncoder()
for col in dataset:
    if dataset[col].dtype == 'object':
        dataset[col] = le.fit_transform(dataset[col])

# DriversLicense(column11) and ZipCode(column13) are not important for predicting credit card approvals. 
# Drop columns 11 and 13
dataset = dataset.drop([11,13], axis=1)
dataset = dataset.values
#Segregate columns and labels into separate variables
x,y = dataset[:,0:13],dataset[:,-1]
#Split the data for training (80%) and testing (20%).
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Use MinMaxScaler to rescale x_train and x_test
sc = preprocessing.MinMaxScaler(feature_range=(0,1))
new_x_train = sc.fit_transform(x_train)
new_x_test = sc.fit_transform(x_test)

#A credit card application will be approved or not is a classification task, 
#so I will use a Logistic Regression model.
model = LogisticRegression()
model.fit(new_x_train, y_train)

# Evaluate the performance of the model on the test set by looking at classification accuracy. 
# And also use The confusion matrix to view model's performance.

#Use model to predict instances from the test set
y_predict = model.predict(new_x_test)
acc_score = model.score(new_x_test, y_test)
print("The accuracy score of Logistic Regression model is ", acc_score)
print(confusion_matrix(y_test, y_predict))

# This is the output,
# The accuracy score of Logistic Regression model is  0.8333333333333334
#[[60 10]
# [13 55]]

## Conclusion
## The model performs well, the accuracy score is 83%.
## For the confusion matrix, 
# the first element which is 60 denotes the true negatives meaning the number of negative instances (denied applications) predicted by the model correctly. 
# And the last element which is 55 denotes the true positives meaning the number of positive instances (approved applications) predicted by the model correctly.
