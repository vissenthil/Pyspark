# https://www.youtube.com/watch?v=-tZbkgTnGs4&list=PL3N9eeOlCrP5PfpYrP6YxMNtt5Hw27ZlO&index=8
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
spark = SparkSession.Builder().appName(name='Test').getOrCreate()

data = spark.read.format('csv').option('mode','DROPMALFORMED').option('header','True')\
    .option('inferSchema','True').load('D:\\python\\PySpark\\DataSets\\LoanStats.csv')
data.printSchema()
null_dropped = data.na.drop(how='any',thresh=20)
null_dropped.show()
print(null_dropped.count())
null_dropped.createOrReplaceTempView('LoanStats')
null_dropped.select(col('member_id'),col('funded_amnt'),col('loan_amnt'),col('total_pymnt')).filter(col('member_id').isNotNull()).show()

def dropNullColumns(df):
    import pyspark.sql.functions as sqlf
    null_counts = df.select([sqlf.count(sqlf.when(sqlf.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()# 1
    col_to_drop = [k for k, v in null_counts.items() if v > 0]  # 2
    print(col_to_drop)
    df = df.drop(*col_to_drop)
    return df


#df = dropNullColumns(null_dropped)
#df.show()
#print(df.columns)
df = null_dropped.select("term","home_ownership","grade","purpose","int_rate","addr_state","loan_status","application_type",
      "loan_amnt","emp_length",
      "annual_inc","dti","delinq_2yrs","revol_util","total_acc","num_tl_90g_dpd_24m","dti_joint")
df.show(10)

import pyspark.sql.functions as F

# Create a dictionary to store the
# count of nulls present in each column.
null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(
	c) for c in df.columns]).collect()[0].asDict()
print('Count of null values in each column:')
print(null_counts)

# Calculate the size of the DataFrame.
df_size = df.count()

# Make a list expression using
# the dictionary to drop columns
# whose value is equal to the data frame size.
to_drop = [k for k, v in null_counts.items() if v == df_size]
# above dictionary will contain all the values null in the column
print('Column to be dropped:')
print(to_drop)

# Drop all the columns present in that list.
#output_df = df.drop(*to_drop)

#output_df.show(truncate=False)
df.cache()
df.select(col("home_ownership"),col("loan_amnt")).describe().show()

df.select('emp_length').show(30)
# Here we are replacing 10+ 2 year 3 years values as only numeric values.
from pyspark.sql.functions import regexp_replace,regexp_extract
regexp_string = 'years|year|\\+|\\<'
df.select(regexp_replace(col("emp_length"),regexp_string, "").alias("emp_length_cleaned"),col("emp_length")).show()

# Another way of doing it. Here we are using regexp_extract note.
regexp_string = '\\d+'
df.select(regexp_extract(col("emp_length"),regexp_string, 0).alias("emp_length_cleaned"),col("emp_length")).show()

# Creating new column in the dataframe previous only for displayed not added into datafram
# here we are creating two new columns
df = df.withColumn("emp_length_cleaned",regexp_extract(col("emp_length"),regexp_string,0))\
    .withColumn("term_cleaned",regexp_replace(col("term"),"months",""))
df.select(col("emp_length"),col("emp_length_cleaned"),col("term"),col("term_cleaned")).show()

print('co-varience :',df.stat.cov('annual_inc','loan_amnt'))
print('co-relation :',df.stat.corr('annual_inc','loan_amnt'))

df.stat.crosstab('loan_status','grade').show(truncate=False)
freq = df.stat.freqItems(['purpose','grade'],0.3)
print(freq.collect())
df.groupBy('purpose').count().show()
df.groupBy('purpose').count().orderBy(col('count').desc()).show()

from pyspark.sql.functions import count,stddev_pop,max,min,avg
quantileProb = [0.25,0.5,0.75,0.0]
relError = 0.5
print(df.stat.approxQuantile('loan_amnt',quantileProb,relError))

relError = 0.0 # No error it will take moee time to process
print(df.stat.approxQuantile('loan_amnt',quantileProb,relError))

quantileProb = [0.25,0.5,0.75,0.0]
relError = 0.5
print(df.stat.approxQuantile('loan_amnt',quantileProb,relError))

from pyspark.sql.functions import isnan,col,count,when

# counting nan and isnull count againt each column in the dataframe
df.select([count(when(isnan(c) | col(c).isNull(),c)).alias(c) for c in df.columns]).show()
#for c in df.columns:
   #if df.count(isnan(c))  : #or df.select(count(when(col(c).isNull(),c))) ) :

#print('Loan status count:',df.filter(df['loan_status'].isNull()).show())

print(df.count())
print(df.groupBy('loan_status').sum('loan_amnt').show(truncate=False))


