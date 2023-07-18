#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score
from sklearn.tree import DecisionTreeClassifier

#read_training data
training_data = pd.read_csv('Train/Train.csv')

#read testing data
test_data = pd.read_csv('Test/Test.csv')
sample_data = pd.read_csv('Test/sample_submission.csv')

#splitting data
y_train = training_data['survived']
x_train = training_data.drop(['survived'],axis=1)

#creating object of LabelEncoder
le = LabelEncoder()

#performing label encoding in training data
x_train['sex']=le.fit_transform(x_train['sex'])
x_train['embarked']=le.fit_transform(x_train['embarked'])
x_train['name']=le.fit_transform(x_train['name'])
x_train['ticket']=le.fit_transform(x_train['ticket'])
x_train['fare']=le.fit_transform(x_train['fare'])
x_train['home.dest']=le.fit_transform(x_train['home.dest'])
x_train['boat']=le.fit_transform(x_train['boat'])
x_train['cabin']=le.fit_transform(x_train['cabin'])

#handling null values in training data
x_train['body'] = x_train['body'].fillna(x_train['body'].mean())
x_train['age'] = x_train['age'].fillna(x_train['age'].mean())

#create object of model
model = DecisionTreeClassifier(criterion='entropy')

#fit training data into the model
model.fit(x_train,y_train)

#perform label encoding on testing data
test_data['sex']=le.fit_transform(test_data['sex'])
test_data['embarked']=le.fit_transform(test_data['embarked'])
test_data['name']=le.fit_transform(test_data['name'])
test_data['ticket']=le.fit_transform(test_data['ticket'])
test_data['fare']=le.fit_transform(test_data['fare'])
test_data['home.dest']=le.fit_transform(test_data['home.dest'])
test_data['boat']=le.fit_transform(test_data['boat'])
test_data['cabin']=le.fit_transform(test_data['cabin'])

#handling null value in testing data
test_data['body'] = test_data['body'].fillna(test_data['body'].mean())
test_data['age'] = test_data['age'].fillna(test_data['age'].mean())

#predicting the values based on the testing data
y_pred = model.predict(test_data)

#finding the error ans accuracy score of model by comparing the 
# #predicted values and sample data that was imported previously
print(r2_score(sample_data['survived'],y_pred))
print(mean_absolute_error(sample_data['survived'],y_pred))
print(accuracy_score(sample_data['survived'],y_pred))