#University admission prediction model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,LassoCV,RidgeCV,ElasticNet,ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
sns.set()

# Function to find out adjected r2
def adj_r2(x,y):

    r2 = regression.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjected_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjected_r2

data = pd.read_csv('D:\\python\\MLProjects\\Admission_Prediction.csv')
print(data.head(20))
print(data.describe(include='all'))
from pandas_profiling import ProfileReport
#pf = ProfileReport(data)
#pf.to_widgets()
#pf.to_file('D:\\python\\MLProjects\\AdmissionProFileRep.html')
print('missing value :',data['University Rating'].isnull().sum())
print('Pre processing of data and filling missing values')
data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['GRE Score'] = data['GRE Score'].fillna(data['GRE Score'].mean())
print('missing value :',data['University Rating'].isnull().sum())
print('Now all the missing values are filled')
print('Srno is not required to creating model so dorping the column')
data = data.drop(columns=['Serial No.'])
print(data.head(20))
print('Now let see how data is distributed every column')
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber <=16:
        ax = plt.subplot(4,4,plotnumber)
        sns.displot(data[column])
        #plt.xlabel(column,fontdict=20)
    plotnumber +=1
    plt.tight_layout()
    #file_name ='distri_'+str(plotnumber)
    #plt.savefig('D:\\python\\MLProjects\\'+file_name+'.png')

print(" let's observe the realationship between independant and dependant variables")
y = data['Chance of Admit'] # dependant variable
x = data.drop(columns=['Chance of Admit']) # independant variable
plotnumber = 1

'''for column in x:
    if plotnumber <=15:
       plt.figure(figsize=(20,30),facecolor='white')
       ax = plt.subplot(5,3,plotnumber)
       plt.scatter(x[column],y)
       plt.xlabel(column)
       plt.ylabel('Change of admit')
    plotnumber +=1
plt.show()'''

scaler = StandardScaler()
x_scalled= scaler.fit_transform(x)
df = pd.DataFrame(x_scalled,columns=['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research'])
print(df.head())

print('To check the multico-linearity')
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = x_scalled
vif = pd.DataFrame() # Creating an empty dataframe
for i in range(variables.shape[1]):
    print(i)
vif['VIF'] = [variance_inflation_factor(variables,i) for i in range(variables.shape[1])]
vif['Features'] = x.columns
print(vif)
x_train,x_test,y_train,y_test = train_test_split(x_scalled,y,test_size=0.25,random_state=355)

regression = LinearRegression()
regression.fit(x_train,y_train)
import pickle  # To store the model in the local file system
filename = 'Finalized_model.pickle'
pickle.dump(regression,open(filename,'wb'))
score = regression.score(x_train,y_train)
print('Accurecy score train :',score)
print('Actual y values ')
print(y_test)
y_hat = regression.predict(x_test)
print('Predicted values:',y_hat)
print('Accurecy score test :',score)
print('adjested r2 value train is:',adj_r2(x_train,y_train))
score = regression.score(x_test,y_test)
print('adjested r2 value test is:',adj_r2(x_test,y_test))
print('Co-efficient values are :',regression.coef_)
print('Intercept values are :',regression.intercept_)
print('Single value prediction using Single row to predict is called sinple value prediction')
singlevalPrediction = regression.predict(np.array([[1.842741e+00,1.788542,0.782010,1.098944,1.776806,0.886405,0.7645085]]))
print('Single value prediction is:',singlevalPrediction)
print('lasso regression')
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit(x_train,y_train)
print('Lasso coefficent :',reg.coef_)
print('Lasso intercept values is :',reg.intercept_)
print('Prediction using test data bulk prediction')
lassoPredcition = reg.predict(x_test)
print('Lasso prediction value is:',lassoPredcition)
print('Lasso single value prediction')
lassoSinplePrediton =reg.predict([[1.842741e+00,1.788542,0.782010,1.098944,1.776806,0.886405,0.7645085]])
print('Lasso Sinple value prediction is:',lassoSinplePrediton)
lassCv = LassoCV(alphas =None,cv =5,normalize =True)
lassCv.fit(x_train,y_train)
alpha = lassCv.alpha_
print('Lasso CV Alpha value is:',alpha)
print('list of alplha values to set in LassoCV')
alphas = np.random.uniform(low=0, high=10, size=(50,))
print(alphas)
LassCV2 = LassoCV(alphas = alphas, cv=5, normalize=True)
LassCV2.fit(x_train,y_train)
alpha = LassCV2.alpha_
print('Best selected aplha value is',alpha)