from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, NGram
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    
    
    sc = SparkContext(appName="SpamDetection")
    sess = SparkSession(sc)
    raw = sess.read.option("delimiter",
                        "\t").csv('/user/edureka_854312/spam_detection/SMSSpamCollection').toDF('spam', 'message')
    trainingData, testData = raw.randomSplit([0.7, 0.3])
    
    tokenizer = Tokenizer().setInputCol('message').setOutputCol('words')
    stopwords = StopWordsRemover().getStopWords() + ['-']
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol('words').setOutputCol('filtered')
    bigram = NGram().setN(2).setInputCol('filtered').setOutputCol('bigrams')
    cvmodel = CountVectorizer().setInputCol('filtered').setOutputCol('features')
    cvmodel_ngram = CountVectorizer().setInputCol('bigrams').setOutputCol('features')
    indexer = StringIndexer().setInputCol('spam').setOutputCol('label')   
    
    nb = NaiveBayes(smoothing=1)
    pipeline = Pipeline(stages = [tokenizer, remover, cvmodel, indexer, nb])
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)
    predictions.select('message', 'label', 'rawPrediction', 'probability', 'prediction').show(5)

    evaluator = BinaryClassificationEvaluator().setLabelCol('label').setRawPredictionCol('prediction').setMetricName('areaUnderROC')
    AUC = evaluator.evaluate(predictions)
    print("AUC:", AUC)

    model.save('/user/edureka_854312/spam_detection/model/')