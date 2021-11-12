package TFIDF

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object TFIDF {
  def main(args: Array[String]): Unit = {
    // Создает сессию спарка
    val spark = SparkSession.builder()
      // адрес мастера
      .master("local[*]")
      // имя приложения в интерфейсе спарка
      .appName("made-tfidf")
      // взять текущий или создать новый
      .getOrCreate()

    // синтаксический сахар для удобной работы со спарк
    import spark.implicits._

    // прочитаем датасет https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/Users/viconstel/IdeaProjects/made-hw4-spark-tfidf/data/tripadvisor_hotel_reviews.csv")
      .withColumn("id", monotonically_increasing_id())

    val dfLower = df
      .withColumn("ReviewLower", lower(col("Review")))

    val dfLowerClean = dfLower
      .withColumn("ReviewLowerClean",
        split(regexp_replace(col("ReviewLower"), "[^0-9a-z ]", ""), " "))

    val dfSentLength = dfLowerClean
      .withColumn("SentLen", size(col("ReviewLowerClean")))

    val dfExploded = dfSentLength
      .withColumn("Word", explode(col("ReviewLowerClean")))
      .filter(col("Word").notEqual(""))

    val dfCountWord = dfExploded
      .groupBy("id", "Word")
      .agg(
        count("Word") as "WordCount",
        first("Sentlen") as "SentLen"
      )

    val dfTf = dfCountWord
      .withColumn("TF", col("WordCount") / col("SentLen"))

    val dfDocCount = dfExploded
      .groupBy("Word")
      .agg(
        countDistinct("id") as "DocCount"
      )

    val dfDocBest = dfDocCount
      .orderBy(desc("DocCount"))
      .limit(100)

    val docAmount = df.count().toDouble
    val computeIdf = udf((value: Int) => math.log(docAmount / value.toDouble))

    val dfIdf = dfDocBest
      .withColumn("IDF", computeIdf(col("DocCount")))

    val dfJoined = dfTf
      .join(dfIdf, Seq("Word"), "inner")
      .withColumn("TfIdf", col("TF") * col("IDF"))

    val dfPivot = dfJoined
      .groupBy("id")
      .pivot(col("Word"))
      .agg(
        first(col("TfIdf"))
      )
      .na.fill(0.0)

    dfPivot.show()
    dfPivot
      .coalesce(1)
      .write
      .option("sep", ",")
      .option("header", "true")
      .csv("/Users/viconstel/IdeaProjects/made-hw4-spark-tfidf/data/top100words_tfidf.csv")
  }
}
