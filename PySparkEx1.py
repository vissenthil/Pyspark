from pyspark import SparkContext
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.Builder().appName(name='Practice').getOrCreate()


#file_paths
flight_perf_path ='D:\\python\\PySpark\DataSets\\departuredelays.csv'
aireports_filepath = 'D:\\python\\PySpark\DataSets\\airport-codes-na.txt'
data = spark.read.format('csv').option('header','True').option('inferSchema','True').load(flight_perf_path)
print(data.show())
airports_data = spark.read.csv(aireports_filepath,header=True,inferSchema=True,sep='\t')
print(airports_data.show())
airports_data.createOrReplaceTempView('airports')
data.createOrReplaceTempView('FlightPerformance')
data.cache()

# Query Sum of Flight Delays by City and Origin Code
# (for Washington State)
spark.sql("""
select a.City, 
f.origin, 
sum(f.delay) as Delays 
from FlightPerformance f 
join airports a 
on a.IATA = f.origin
where a.State = 'WA'
group by a.City, f.origin
order by sum(f.delay) desc"""
).show()

