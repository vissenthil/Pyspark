#import SparkSeccion pyspark.sql
from pyspark.sql import SparkSession

#Create my_spark
spark = SparkSession.builder.getOrCreate()

#print my_spark
print(spark)
file_path = 'D:\\python\\PySpark\\DataSets\\airports.csv'
#Read in the airports path
airports = spark.read.csv(file_path, header=True)

airports.show()
flights = spark.read.csv('D:\\python\\PySpark\\DataSets\\flights_small.csv', header=True)
flights.show()
#Filter flights with a SQL string
long_flights1 = flights.filter('distance > 1000')
long_flights1.show()

# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000 )
long_flights2.show()

# Select the first set of columns as a string
selected_1 = flights.select('tailnum', 'origin', 'dest')
# Select the second set of columns usinf df.col_name
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter to only keep flights from SEA to PDX.
FilterA = flights.origin == 'SEA'
FilterB =flights.dest == 'PDX'

# Filter the data, first by filterA then by filterB
selected_2 = temp.filter(FilterA).filter(FilterB)
selected_2.show()

#Create a table of the average speed of each flight both ways.
#Calculate average speed by dividing the distance by the air_time (converted to hours).Use the .alias() method name
# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")
speed_1 = flights.select('origin','dest','tailnum', avg_speed)
speed_1.show(30)

#Using the Spark DataFrame method .selectExpr()
speed_2 =flights.selectExpr('origin','dest','tailnum','distance/(air_time/60) as avg_speed')
speed_2.show()

#arr_time: string and distance: string, so to find min() and max() we need to convert this float
flights = flights.withColumn('distance', flights.distance.cast('float'))
flights = flights.withColumn('air_time', flights.air_time.cast('float'))

flights.describe('air_time', 'distance').show()

#Find the length of the shortest (in terms of distance) flight that left PDX
flights.filter(flights.origin =='PDX').groupBy().min('distance').show()
#Find the length of the longest (in terms of time) flight that left SEA
flights.filter(flights.origin == 'SEA').groupBy().max('air_time').show()
#get the average air time of Delta Airlines flights  that left SEA.
flights.filter(flights.carrier == 'DL').filter(flights.origin == 'SEA').groupBy().avg('air_time').show()

#get the total number of hours all planes in this dataset spent in the air by creating a column called duration_hrs
flights.withColumn('duration_hrs', flights.air_time/60).groupBy().sum('duration_hrs').show()

#Group by tailnum column
by_plane = flights.groupBy('tailnum')
#Use the .count() method with no arguments to count the number of flights each plane made
by_plane.count().show()

#Group by tailnum column
by_plane = flights.groupBy('tailnum')
#Use the .count() method with no arguments to count the number of flights each plane made
by_plane.count().show()

#Group by tailnum column
by_plane = flights.groupBy('tailnum')
#Use the .count() method with no arguments to count the number of flights each plane made
by_plane.count().show()
# Group by month and dest
flights = flights.withColumn('dep_delay', flights.dep_delay.cast('float'))
by_month_dest = flights.groupBy('month', 'dest')
# Average departure delay by month and destination

by_month_dest.avg('dep_delay').show()
airports.show()

# Rename the faa column
airports = airports.withColumnRenamed('faa','dest')
# Join the DataFrames
flights_with_airports= flights.join(airports, on='dest', how='leftouter')
flights_with_airports.show()
planes = spark.read.csv('D:\\python\\PySpark\\DataSets\\planes.csv', header=True)
planes.show()

# Rename year column on panes to avoid duplicate column name
planes = planes.withColumnRenamed('year', 'plane_year')
#join the flights and plane table use key as tailnum column
model_data = flights.join(planes, on='tailnum', how='leftouter')
model_data.show()
model_data.describe()
model_data = model_data.withColumn('arr_delay', model_data.arr_delay.cast('integer'))
model_data = model_data.withColumn('air_time' , model_data.air_time.cast('integer'))
model_data = model_data.withColumn('month', model_data.month.cast('integer'))
model_data = model_data.withColumn('plane_year', model_data.plane_year.cast('integer'))
model_data.describe('arr_delay', 'air_time','month', 'plane_year').show()

model_data = model_data.withColumn('arr_delay', model_data.arr_delay.cast('integer'))
model_data = model_data.withColumn('air_time' , model_data.air_time.cast('integer'))
model_data = model_data.withColumn('month', model_data.month.cast('integer'))
model_data = model_data.withColumn('plane_year', model_data.plane_year.cast('integer'))
model_data.describe('arr_delay', 'air_time','month', 'plane_year').show()
# Create a new column
model_data =model_data.withColumn('plane_age', model_data.year - model_data.plane_year)
model_data = model_data.withColumn('is_late', model_data.arr_delay >0)

model_data = model_data.withColumn('label', model_data.is_late.cast('integer'))

model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")
from pyspark.ml.feature import StringIndexer, OneHotEncoder

#Create a StringIndexer
carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')
#Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carr_fact')

# encode the dest column just like you did above
dest_indexer = StringIndexer(inputCol='dest', outputCol='dest_index')
dest_encoder = OneHotEncoder(inputCol='dest_index', outputCol='dest_fact')
# Assemble a  Vector
from pyspark.ml.feature import  VectorAssembler
vec_assembler =VectorAssembler(inputCols=['month', 'air_time','carr_fact','dest_fact','plane_age'],
                              outputCol='features',handleInvalid="skip")


# #### Create the pipeline
# You're finally ready to create a` Pipeline!` Pipeline is a class in the `pyspark.ml module` that combines all the Estimators and Transformers that you've already created.

from pyspark.ml import Pipeline

flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

piped_data =flights_pipe.fit(model_data).transform(model_data)
piped_data.show()

training, test = piped_data.randomSplit([.6, .4])
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()
# #### Create the evaluator
# The first thing you need when doing cross validation for model selection is a way to compare different models. Luckily, the pyspark.ml.evaluation submodule has classes for evaluating different kinds of models. Your model is a binary classification model, so you'll be using the `BinaryClassificationEvaluator` from the `pyspark.ml.evaluation` module. This evaluator calculates the area under the ROC. This is a metric that combines the two kinds of errors a binary classifier can make (false positives and false negatives) into a simple number.

import pyspark.ml.evaluation as evals

evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()
import numpy as np
# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()
# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
                         estimatorParamMaps=grid,
                         evaluator=evaluator
                         )
# Fit cross validation models
models = cv.fit(training)
# Extract the best model
best_lr = models.bestModel
# Use the model to predict the test set
test_results = best_lr.transform(test)
#
# Evaluate the predictions
print(evaluator.evaluate(test_results))
