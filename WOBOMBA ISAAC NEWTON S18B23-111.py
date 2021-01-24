import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Get the stock data using amazon's data set
df=quandl.get("WIKI/AMZN")
#Take a look at the data
print(df.head())

#Get the Adjusted Close Price
df=df[['Adj. Close']]
#Take a look at the new data
print(df.head())

#Variable for predicting 30 days out into the future
ft_out=30
#Create another colum(The target for the dependent variable shifted in units)
df['Prediction']=df[['Adj. Close']].shift(-ft_out)
#print the new data set
print(df.tail())

#Create the independent data set (a)
#converting the data frame to a numpy array
a=np.array(df.drop(['Prediction'],1))
#Removing the last 'n' rows
a=a[:-ft_out]
print (a)

#Create the dependent data set(b)
#convert the data frame to a numpy array(All of the values including the NaN's)
b=np.array(df['Prediction'])
#Getting all the b values except the last 'n' rows
b=b[:-ft_out]
print (b)

#Splitting the data into 80% training and 20% testing
a_train, a_test, b_train, b_test=train_test_split(a,b,test_size=0.2)

#Create and train the support vector machine(regressor)
svr_rbf=SVR(kernel='rbf',C=1e3, gamma=0.1)
svr_rbf.fit(a_train, b_train)

#Testing model: score returns the coefficient of determination R^2 of the prediction
#the best possible score is 1.0
svm_confidence =svr_rbf.score(a_test, b_test)
print('svm confidence: ',svm_confidence)

#Create and train the Linear Regression Model
lr=LinearRegression()
#Train the model
lr.fit(a_train, b_train)

#Testing Model:Score returns the coefficient of determination R^2 of the prediction
#The best possible is 1.0
lr_confidence=lr.score(a_test, b_test)
print('lr confidence: ', lr_confidence)

#Set a_forecast equal to the last 30 rows of the original data set from the Adj. Close Column
a_forecast=np.array(df.drop(['Prediction'], 1))[-ft_out:]
print (a_forecast)

#Print the linear regression model predictions for the next 'n' days
lr_prediction=lr.predict(a_forecast)
print(lr_prediction)

#Print the support regressor model predictions for the next 'n' days
svm_prediction=svr_rbf.predict(a_forecast)
print(svm_prediction)