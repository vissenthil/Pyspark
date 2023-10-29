import os
import sys
from pyspark.sql import SparkSession
# To avoid Python 3 not found error. Need to set System environment variable
#as PYSPARK_HOME = python  and below two line of code to void this error.

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from pyspark.sql import SparkSession,SQLContext

from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import *
spark = SparkSession.Builder().appName('Datafrmae Example').getOrCreate()


data2 = [("James", "", "Smith", "36636", "M", 3000),
         ("Michael", "Rose", "", "40288", "M", 4000),
         ("Robert", "", "Williams", "42114", "M", 4000),
         ("Maria", "Anne", "Jones", "39192", "F", 4000),
         ("Jen", "Mary", "Brown", "", "F", -1)
         ]

schema = StructType([
    StructField("firstname", StringType(), True),
    StructField("middlename", StringType(), True),
    StructField("lastname", StringType(), True),
    StructField("id", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("salary", IntegerType(), True)
    ])

df = spark.createDataFrame(data=data2, schema=schema)
df.printSchema()
df.select(col('firstname')).show()
df.groupBy('gender').agg(sum('salary').alias('sum_sal')).orderBy(asc('sum_sal')).show()

simpleData = [("James","Sales","NY",90000,34,10000),
    ("Michael","Sales","NV",86000,56,20000),
    ("Robert","Sales","CA",81000,30,23000),
    ("Maria","Finance","CA",90000,24,23000),
    ("Raman","Finance","DE",99000,40,24000),
    ("Scott","Finance","NY",83000,36,19000),
    ("Jen","Finance","NY",79000,53,15000),
    ("Jeff","Marketing","NV",80000,25,18000),
    ("Kumar","Marketing","NJ",91000,50,21000)
  ]

schema = ["employee_name","department","state","salary","age","bonus"]
df = spark.createDataFrame(data=simpleData, schema = schema)
df.printSchema()
df.show(truncate=False)

df.groupBy('department').sum(col('salary').desc()).show()
