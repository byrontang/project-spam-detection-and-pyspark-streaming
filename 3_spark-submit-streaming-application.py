from __future__ import print_function

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
import datetime

if __name__ == "__main__":
    
    # Spark Streaming
    sc = SparkContext(appName="SpamDetection")
    ssc = StreamingContext(sc, 60)
    
    now = datetime.datetime.now()
    filepath = "/user/edureka_854312/spam_detection/" + now.strftime("%Y-%m-%d/")
    print("filepath:", filepath)
    lines = ssc.textFileStream(filepath)
    
    def process(t, rdd):
        if rdd.isEmpty():
            print("filepath:", filepath)
            print("==== EMPTY ====")
            return

        print("=== RDD Found ===")
        spark = SparkSession.builder.getOrCreate()
        rowRdd = rdd.map(lambda x: Row(message=x))
        df = spark.createDataFrame(rowRdd)
        print("=== DataFrame ===")
        print(df.show())
        
        if not rdd.isEmpty():
            model = PipelineModel.load('/user/edureka_854312/spam_detection/model/')
            
            # Predict the SPAM messages and print the SPAM in the logs
            predictions = model.transform(df)
            print("=== Prediction ===")
            print(predictions.show())
    
    lines.pprint()  
    lines.foreachRDD(process)
    
    ssc.start()
    ssc.awaitTermination()