from pyspark import SparkContext, SparkConf

DATA = "./data/*.txt"
OUTPUT_DIR = "counts" # name of the folder

def word_count():
    conf = SparkConf().setAppName("Word Count Larger Than 3").setMaster("local")
    sc = SparkContext("local","Word count example")
    textFile = sc.textFile(DATA)
    counts = textFile.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    filtered_counts = counts.filter(lambda pair: pair[1] >= 3)
    filtered_counts.saveAsTextFile(OUTPUT_DIR)
    print("words that have count greater or equal to 3 saved")
    sc.stop()
word_count()
