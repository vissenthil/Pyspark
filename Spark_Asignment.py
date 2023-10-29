from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, udf, min, max, concat, column, row_number
from pyspark.sql.window import WindowSpec, Window
import os
from pyspark.sql.functions import explode
from pyspark.sql.functions import flatten
from pyspark import SparkConf
from pyspark.sql import SparkSession, Row


def spark_json():
    spark = SparkSession.builder.appName("spark_test").getOrCreate()
    sc = spark.sparkContext

    sc.setLogLevel("Error")

    print("Dataframe reader:")

    # os.system("hadoop fs -put /home/Raj/data/arxiv-metadata-oai-snapshot.json /tmp/")
    # df = spark.read.json("/mystorage_demo/arxiv-metadata-oai-snapshot.json")
    df = spark.read.json("gs://mystorage_demo/arxiv-metadata-oai-snapshot.json")
    df.printSchema()

    print(df.rdd.getNumPartitions())

    Schema = StructType([
        StructField('authors', StringType(), True),
        StructField('categories', StringType(), True),
        StructField('license', StringType(), True),
        StructField('comments', StringType(), True),
        StructField('abstract', StringType(), True),
        StructField('versions', ArrayType(StringType()), True),
    ])

    df1 = spark.read.json("gs://mystorage_demo/arxiv-metadata-oai-snapshot.json", schema=Schema)

    # df_noNull=df1.na.drop("all")

    print("check for null values in a column")
    df1.filter(df1.comments.isNull()).show()

    print("dropping the records containing null values")
    df2 = df1.dropna(subset=["comments"])

    print("replace null values with unknown keyword")
    df3 = df2.fillna(value="unknown", subset=["license"])

    df3.show()

    df3.select(col("versions")).show(1, False)

    print("flatten a json df")

    flatten_df = df3.select(df3.authors, explode(df3.versions))
    flatten_df.show(truncate=False)

    flatten_df.write.mode('overwrite').parquet("hdfs:///result/flatten_author_versions")

    df3.createOrReplaceTempView("Archive")
    sql_query = """ SELECT distinct authors FROM Archive
                WHERE categories LIKE 'math%'
                """
    authers_list = spark.sql(sql_query)
    print(spark.sql(sql_query).count())

    authers_list.write.mode('overwrite').parquet("hdfs:///result/authers_list")


if __name__ == '__main__':
    spark_json()