import pandas as pd    #read dataset from excel file

import numpy    #math calc

import warnings

warnings.filterwarnings('ignore')  #ignore warnings


#1.data collection

dataset = pd.read_csv('hearts.csv')   #read from hearts.csv file

#print(dataset)  #end


#2.data preprocessing

from sklearn.preprocessing import LabelEncoder   #preprocessing library

le = LabelEncoder()

dataset['Sex'] = le.fit_transform(dataset['Sex'])     #taking Sex column which is alphabetical and converting to numerical data

dataset['ChestPainType'] = le.fit_transform(dataset['ChestPainType'])  #taking chest pain type column which is alphabetical and converting to numerical data

dataset['RestingECG'] = le.fit_transform(dataset['RestingECG'])     #taking resting ecg column which is alphabetical and converting to numerical data

dataset['ExerciseAngina'] = le.fit_transform(dataset['ExerciseAngina']) #taking exercise angina column which is alphabetical and converting to numerical data

dataset['ST_Slope'] = le.fit_transform(dataset['ST_Slope'])     #taking st slope column which is alphabetical and converting to numerical data

##print(dataset)   #end


x = dataset.drop(columns = ['HeartDisease'])   #dropping heart disease col which is op column, taking only ip cols

y = dataset['HeartDisease']   #op column



#Spliting dataset for training and testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 12)   #20 % for testing, randomly shuffle data

#print(x_train.shape)    #print count of rows and cols



#3.model training

from sklearn.naive_bayes import GaussianNB   #naive bayes algorithm

nb = GaussianNB()

nb.fit(x_train,y_train)   #training ip and op

print("training completed")  #end


#4.model evaluation

y_pred = nb.predict(x_test)  #predict ip testing data and store op in y_pred

##print("y pred",y_pred)
##
##print("y test",y_test)

from sklearn.metrics import accuracy_score   #see accuracy

print("Accuracy is ", accuracy_score(y_test,y_pred))


#5.Model prediction

test = nb.predict([[19,1,4,120,166,0,1,138,0,0,2]])   #giving random input for 11 cols(refer x_train ip)

if test ==1:

    print("Heart disease detected")

else:

    print("Normal")









