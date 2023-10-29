import pandas as pd
import numpy as np

print('Creating empty DataFrame')
df = pd.DataFrame()
print(df)
print('Sample Serios')
ls =['senthil','Thalir','Aruthra','Kruthikshashini','our family']
df = pd.Series(ls)
print(df)
print('Sample 1 creating dataframe using dictionary')
data ={'playeser':['Senthil','Thalir','Aruthra','Kruthi'],'Matches':[100,200,300,500]}
df = pd.DataFrame(data)
print(df)
print(df.loc[1]) # iS A Index Location

print('DataFrame idex')
df = pd.DataFrame({'Name':['Senthil','Thalir','Aruthra','Kruthi'],
                          'Age':[45,34,6,3],
                          'Location':['New yark','Chennai','Malaysia','Kana']},
                      index=([10,20,40,60]))
print(df)

print(df.loc[10:40])
print('Displaying dataframe columns:')
print(df.columns)
print('Dispaying index in DataFrame')
print(df.index)
print(''' This method prints information about a DataFrame including the index 
           dtype and columns, non-null values and memory usage''')
print(df.info())
print(''' Return a subset of the DataFrame’s columns based on the column dtypes.''')
print('''DataFrame.select_dtypes(include=None, exclude=None)[source]
 ''')
print('Selecting int datatypes')
print(df.select_dtypes(include='int64'))
print('Selecting float types')
print(df.select_dtypes(include='float64'))


print('Sample 2')
row = ['X','Y','Z','S']
cols = ['A','B','C','D','E']
data = np.round(np.random.randn(4,5),2)
print(data)
df = pd.DataFrame(data,row,cols)
print(df)
print(df[['B','A']])
print(df['A']['X'])
print(df['A']['Y'])
print('Creating new column and removing a column')
df['A+B'] = df['A'] + df['B']
print(df)
df.drop('A+B',inplace=True,axis=1)
print(df)
print('Droping a rows')
df.drop('S',inplace=True)
print(df)
print('Displaying shape of the dataframe')
print(df.shape)

# Create first Dataframe using dictionary
info1 = pd.DataFrame({"x":[25,15,12,19],"y":[47, 24, 17, 29]})
# Create second Dataframe using dictionary
Info2 = pd.DataFrame({"x":[25, 15, 12],"y":[47, 24, 17],"z":[38, 12, 45]})
print(info1)
print(Info2)
# append info2 at end in info1
#final_df=info1.concat(Info2, ignore_index = True)
final_df = pd.concat([info1,Info2])
print('Final info after Concate')
print(final_df)
print('Creating New DataFrame info')
info = pd.DataFrame([[2, 3,6]] * 5, columns=['P', 'Q','R'])
print('Apply Sqrt')
info.apply(np.sqrt)
print(info)
print('Droping a row in the dataframe')
print(info.drop([1,3],axis=0))
print('Sum axis=0 means rowise sum')
t1=info.apply(np.sum, axis=0)
print(t1)
print('Sum axis=0 means columnise sum')
t2=info.apply(np.sum, axis=1)
print(t2)