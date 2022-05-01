#importing library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection
#reading csv file into dataframe
diabetes_dataset = pd.read_csv("D:\diabetes.csv")
diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.isnull().sum()
diabetes_dataset.describe()

r,c=diabetes_dataset['Outcome'].value_counts()
print(r,c)

#data visualization
#plotting bar graph for depandent variable
x=["non diabeties","diabeties"]
y=[r,c]
plt.bar(x,y,color=["red","green"])
plt.show()
a=plt.pie([r,c],labels=["no diabeties","diabeties"],autopct="%0.2f%%")
diabetes_dataset.groupby('Outcome').mean()


#dividing the data
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)


#data preprocessing
#standardization of data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = diabetes_dataset['Outcome']


#splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


#model selection
var = svm.SVC(kernel='linear')
var.fit(X_train, Y_train)


#model testing
X_train_prediction = var.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = var.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)




# model deployement
# making a predictive system
input_data = []
i = 0
while (i < 1):
    lis1 = input("enter the no of pregnancies")
    input_data.append(lis1)
    lis2 = input("enter glucose level")
    input_data.append(lis2)
    lis3 = input("enter blood pressure level")
    input_data.append(lis3)
    lis4 = input("skin thickness")
    input_data.append(lis4)
    lis5 = input("enter insulin level")
    input_data.append(lis5)
    lis6 = input("enter BMI")
    input_data.append(lis6)
    lis7 = input("enter dibeties pedegree function")
    input_data.append(lis7)
    lis8 = input("enter age")
    input_data.append(lis8)
    i = i + 1

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = var.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')


