from pyspark import SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.Builder().appName(name='Practice').getOrCreate()

print(spark)
df_spark = spark.read.csv('D:\\python\\MLProjects\\Admission_Prediction.csv')

print(df_spark.head(4))
print(df_spark.show())
# This is to consider the first row as the header
df_spark = spark.read.option('header',True).csv('D:\\python\\MLProjects\\Admission_Prediction.csv',inferSchema=True)
print(df_spark.show())
print('To get the data type of the dataframe same like pandas df.info()')
#Check the schema and check the datatype of the schema
print(df_spark.printSchema())
print(type(df_spark))
#Check the columns in the schema
print(df_spark.columns)
print('\n')
print('\n',df_spark.head(5))
# hOW to select particular columns in the dataframe
print('\n',df_spark.select('University Rating').show())
# How to select multiple columns
print('\n', df_spark.select(['TOEFL Score','University Rating']).show())
# How to check the datatype
print(df_spark.dtypes)
#How to use descripe as like in pandas
print(df_spark.describe().show())
# How to add columns in the data frame
# Here New University Rating is the new colum and University Rating is old column adding with 2
df_spark = df_spark.withColumn('New University Rating',df_spark['University Rating']+2)
print(df_spark.select(['New University Rating','University Rating']).show())
# How to drop the columns
df_spark = df_spark.drop('New University Rating')
print('Printing columns list after drop')
print(df_spark.columns)
# How to rename a column TOEFL Score is the old column name and Toefl score is new column name
df_spark = df_spark.withColumnRenamed('TOEFL Score','Toefl score')
print(df_spark.columns)
# spark handling missing values.
print('''--------------Sprack handling missing values --------------------
            1. Droping columns
            2. Droping rows
            3. Various parameter in droping functionality
            4. Handling missing values by mean, median and mode
''')

df_spark = df_spark.drop('Serial No.')
print(df_spark.show())
print('Drop the null rows in the datafrmae')
df_spark = df_spark.na.drop()
print(df_spark.show())
df_spark = df_spark.dropna(how='all') # this will drop if all the values in the rows are null
# if any values without null in the row it will not drop that row. All values should be null

df_spark = df_spark.dropna(how='any') # this will drop if any  values in the rows are null\
#Threshold
df_spark = df_spark.dropna(how='any',thresh=2) # It means atleast two non null values present
# it will not delete if only it has altest 2 non null values present. if one means it will delete.

#subset means if only given columns if null values are there it will drop those rows.
df_spark = df_spark.dropna(how='any',subset=['University Rating'])
print('Filling the missing values')
df_spark = df_spark.fillna(1,['Research','Toefl score']) # parameters 1.values, 2.subset
# Here 1 is replaced in Research, Toefl score columns if it has null values.

from pyspark.ml.feature import Imputer
imputer = Imputer(
    inputCols=['GRE Score','Toefl score','Research'],
    outputCols=["{}_imputed".format(c) for c in ['GRE Score','Toefl score','Research']]
).setStrategy('mean') # here we can apply mean or median or mode as parameter.

# Add imputation columns in the data frame.
imputed = imputer.fit(df_spark).transform(df_spark)
print(imputed.show())

print('''--------------------Spark filter operations ---------------------------
          & ,|, ==, ~ ''')

print(df_spark.filter(df_spark['GRE Score']>=325).show())
print(df_spark.filter(df_spark['GRE Score']<300).show())
# Below one will select University Rating and SOP Columns which has GRE Score less then 3000
print(df_spark.filter(df_spark['GRE Score']<300).select(['University Rating','SOP']).show())
# using AND Conditions same way we can use OR Condition also.
print(df_spark.filter((df_spark['GRE Score']<=300) & (df_spark['Toefl score'] >=103)).show())
# using NOT In pyspark is ~ simbal, Here GRE Score is not less then 300 will be shown
# means greater than 300 will be shown.
print(df_spark.filter(~(df_spark['GRE Score']<300)).show())

print(' -----------------PySpark Groupby and Aggregate functions --------------------')
#spark.stop()
data = spark.read.csv('D:\\python\\MLProjects\\PySparKDemo.csv',header=True,inferSchema=True)
print(data.show())
print(data.groupBy('Name').sum().show())
# Group by to get the maximum salary department wise
print(data.groupBy('Department').sum().show())
# To get the mean
print(data.groupBy('Department').mean().show())

# To get count of people working in the department
print(data.groupBy('Department').count().show())
# aggregate function with key value pare it will provide sum of total salary
print(data.agg({'Salary':'sum'}).show())
# How to get the department wise maximum Salary
print('Maximum Salary Department wise\n')
print(data.groupBy('Department').max().show())
print('Minum Salary Department wise\n')
print(data.groupBy('Department').min().show())
print('Average Salary Department wise\n')
print(data.groupBy('Department').avg().show())

#PySpark ML
print('----------------------PySpark ML ------------------------------------------')
# inferSchema Without this all datatype will be string if we use this one based on values it
# will take it as string or integer
data_ML = spark.read.csv('D:\\python\\MLProjects\\PySparkML.csv',header=True,inferSchema=True)
print(data_ML.show())
print(data_ML.printSchema())
print(data_ML.columns)
# importing modules
from pyspark.ml.feature import VectorAssembler
# Here independent feature is grouped into new feature as Independent feature as shown below
#in outputcol
featureAssembler = VectorAssembler(inputCols=['Age','Experience'],outputCol='Independent feature')
# Below is transform the grouped data into output dataframe
output = featureAssembler.transform(data_ML)
print(output.show())
print(output.columns)
finalized_data = output.select(['Independent feature','Salary'])
#Here Independent feature is x and Salary is y label
print(finalized_data.show())
# Next step is to tran test split
from pyspark.ml.regression import LinearRegression
#train test split.
train_data,test_data = finalized_data.randomSplit([0.75,0.25])
print(test_data.show())
lr = LinearRegression(featuresCol='Independent feature',labelCol='Salary')
lr = lr.fit(train_data)
print(lr.coefficients)
print(lr.intercept)
# Now Prediction
Prediction_result = lr.evaluate(train_data)
print(Prediction_result.predictions.show())
print('meanAbsoluteError:\n')
print(Prediction_result.meanAbsoluteError)
print('meanSquaredError')
print(Prediction_result.meanSquaredError)



