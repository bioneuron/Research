from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from  pyspark.sql.types import *


conf = SparkConf().setAppName('pubmed_open_access').setMaster('local[32]')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


#df = sqlContext.read.load(FILE_NAME)
FILE_NAME = '/home/rasoul/Dropbox/Programming/SEIR/Matlab/data/new/2007_08.csv'
RDD_DATA = sc.textFile(FILE_NAME) \
    .map(lambda line: line.split(",")) \
 #   .map(lambda line: [line(0)]) \
  #  .collect()

#TODO: Test one
x1 = sqlContext.createDataFrame(RDD_DATA)

INNFECTION = x1.select('_3')
HUMIDITY = x1.select('_5')

print(HUMIDITY.count())
data = x1.select('_3', '_5')
data.show()

#TODO: do the query here

#data_new2 = data[data._10 == ""]




