
from pyspark.sql import SparkSession
spark = SparkSession.Builder().appName('Test').getOrCreate()
df = spark.read.format('csv').option('inferSchema','True').option('mode','DROPMALFORMED').option('header','True').load('D:\\python\\PySpark\\DataSets\\spark_EDA_Example.csv')
df.show()
# Dropping a column
#df_dropped = df.na.drop(how='any',subset=['age'])
# subset means in a particular column having null values remove that record.
print(df.columns)
#df_dropped.show()
#df_dropped = df.na.drop(how='any',thresh=3)
#df_dropped.show()
# here thresh =3 means at least 3 nonnull values should be there otherwise it will drop
#df_null_filled = df_dropped.na.fill('Missing value')

#df_dropped.show()
# filling with mean vlaues.
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

#Convert String to Integer Type
#df_dropped.withColumn("experience",df_dropped.experience.cast('int')).printSchema()
from pyspark.ml.feature import Imputer
#df_dropped.printSchema()
imputer = Imputer(
    inputCols = ['age', 'Salary'],
    outputCols = ["{}_imputed".format(a) for a in ['age',  'Salary']]
).setStrategy("mean")

imputer.fit(df).transform(df).show()