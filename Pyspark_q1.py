print('How to identify continuous variables in a dataframe and create a list of those column names?')

from pyspark import SparkContext,SparkFiles
from pyspark.sql import SparkSession
spark = SparkSession.Builder().appName(name='Practice').getOrCreate()


#url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling_m.csv"
#spark.sparkContext.addFile(url)

#df = spark.read.csv(SparkFiles.get("Churn_Modelling_m.csv"), header=True, inferSchema=True)

#df = spark.read.csv("C:/Users/RajeshVaddi/Documents/MLPlus/DataSets/Churn_Modelling_m.csv", header=True, inferSchema=True)
df = spark.read.format('csv').option('header',True).option('inferSchema',True).option('sep',',').load('D:\\python\\PySpark\\DataSets\\Chrun_Modelling_m.txt')
print(type(df))
df.printSchema()
df.show(2, truncate=False)
df.select('CreditScore').show()
print(df.columns)

from pyspark.sql.types import IntegerType, StringType, NumericType
from pyspark.sql.functions import approxCountDistinct
class separate_column_list:
    def __init__(self,df,distinct_threshold):
        self.df = df
        self.distinct_threshold = distinct_threshold
    def detect_continuous_variables(self):
        """
            Identify continuous variables in a PySpark DataFrame.
            :param df: The input PySpark DataFrame
            :param distinct_threshold: Threshold to qualify as continuous variables - Count of distinct values > distinct_threshold
            :return: A List containing names of continuous variables
        """
        continuous_columns = []
        for column in self.df.columns:
            dtype = self.df.schema[column].dataType
            if isinstance(dtype, (IntegerType, NumericType)):
                print(f'columns:{column} = {dtype}')
                distinct_count = self.df.select(approxCountDistinct(column)).collect()[0][0]
                print('count:',distinct_count)
                if distinct_count > self.distinct_threshold:
                  continuous_columns.append(column)
        return continuous_columns


    def String_column_list(self):
        string_column_list =[]
        for column in self.df.columns:
            dtype = self.df.schema[column].dataType
            print(f'columns:{column} = {dtype}')
            if isinstance(dtype,(StringType)):
                string_column_list.append(column)
        return string_column_list



if __name__ == '__main__':
   sepcolumn = separate_column_list(df,10)
continuous_variables = sepcolumn.detect_continuous_variables()  # 9000 is number of distinct values in that column
print(continuous_variables)

print('Printing String Column List')
StringColumnList = sepcolumn.String_column_list()
print(StringColumnList)
df.select('Surname').show()


