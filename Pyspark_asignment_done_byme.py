from pyspark import SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, udf, min, max, concat, column, row_number, avg, count
from pyspark.sql.window import WindowSpec, Window


def spark_test():
    spark = SparkSession.builder.appName("spark_test").getOrCreate()
    sc = spark.sparkContext

    sc.setLogLevel("Error")
    # df = spark.read.json("gs://mystorage_demo/arxiv-metadata-oai-snapshot.json")
    df = spark.read.option('mode', 'DROPMALFORMED').json("gs://mystorage_demo/arxiv-metadata-oai-snapshot.json")

    df.createOrReplaceTempTable('arxiv_snapshot_table')

    df.printSchema()
    df.show(10, False)

    df.select("city", "country", "state", "zipcode").show(10, False)
    df2 = df.select(col("City"), col("Country"), col("EstimatedPopulation")).where(
        col("EstimatedPopulation").isNotNull())

    df2.createOrReplaceTempTable(df2_temp_table)

    spark.sql(""" select * from df2_temp_table;""")

    # overwrite/append
    df2.write.mode('overwrite').saveAsTable("loc.population_by_area")
    df4 = spark.sql("select * from loc.population_by_area")
    df4.show()
    df4.printSchema()


if __name__ == '__main__':
    spark_test()

