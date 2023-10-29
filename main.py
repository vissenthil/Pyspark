import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ipywidgets
from pandas_profiling import ProfileReport
data = pd.read_csv('D:\\python\\MLProjects\\Advertising.csv')

print(data.head(10))
descripe = data.describe()
print(descripe)
profile = ProfileReport(data)
profile.to_file("D:\\python\\MLProjects\\profile.html")

#print(profile.to_widgets())

#profile.to_file("D:\\python\\MLProjects\\widgets.html")
x = data[['TV']]
print(x)
print('Sales data')
y = data.sales
print(y)
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x,y)
intercept = linear.intercept_
print('Intercept value is:',intercept) # yy = mx+c here intercept is the c value
print('Co-relation vlaue is:',linear.coef_) # this is m value
file = 'linear_reg.sav'
pickle.dump(linear,open(file,'wb')) # Created the binary file
predicted_values = linear.predict([[55]])
print(predicted_values)
new_val_for_predict = [34,56,78,12,78,23]
print('Predicted values for New Sets:')
for i in new_val_for_predict:
    print(linear.predict([[i]]))
print('Adding features to find the prediction')
x1 = data[['TV','newspaper','radio']]
print(x1.head(10))
lr2 = LinearRegression()
lr2.fit(x1,y)
print(f'intercept value of lr2 is: {lr2.intercept_}')
print(f'coefficient value of lr2 is: {lr2.coef_}')
predict_with_3features = [[230.1,37.8,69.2]]
prediction_with3features = lr2.predict(x1)
print('New predicted values for TV and newspaer is:',lr2.predict(predict_with_3features))
print('count :',len(prediction_with3features))
print('New predicted values for TV and newspaer and radio is:',prediction_with3features)
