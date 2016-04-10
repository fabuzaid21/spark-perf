package mllib.perf

import mllib.perf.util.{DataGenerator, DataLoader}
import org.apache.spark.SparkContext
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonAST._
import org.json4s.JsonDSL._

/** Parent class for tests which run on a large dataset. */
abstract class RegressionAndClassificationTests[M](sc: SparkContext) extends PerfTest {

  def runTest(rdd: DataFrame): M

  def validate(model: M, rdd: RDD[LabeledPoint]): Double

  val NUM_EXAMPLES =  ("num-examples",   "number of examples for regression tests")
  val NUM_FEATURES =  ("num-features",   "number of features of each example for regression tests")

  intOptions = intOptions ++ Seq(NUM_FEATURES)
  longOptions = Seq(NUM_EXAMPLES)

  var rdd: RDD[LabeledPoint] = _
  var testRdd: RDD[LabeledPoint] = _

  /**
   * For classification
   * @param predictions RDD over (prediction, truth) for each instance
   * @return Percent correctly classified
   */
  def calculateAccuracy(predictions: RDD[(Double, Double)], numExamples: Long): Double = {
    predictions.map{case (pred, label) =>
      if (pred == label) 1.0 else 0.0
    }.sum() * 100.0 / numExamples
  }

  /**
   * For regression
   * @param predictions RDD over (prediction, truth) for each instance
   * @return Root mean squared error (RMSE)
   */
  def calculateRMSE(predictions: RDD[(Double, Double)], numExamples: Long): Double = {
    val error = predictions.map{ case (pred, label) =>
      (pred - label) * (pred - label)
    }.sum()
    math.sqrt(error / numExamples)
  }
}


// Decision-tree
sealed trait TreeBasedModel
case class MLDTRegressionModel(model: DecisionTreeRegressionModel) extends TreeBasedModel
case class MLDTClassificationModel(model: DecisionTreeClassificationModel) extends TreeBasedModel
case class MLRFRegressionModel(model: RandomForestRegressionModel) extends TreeBasedModel
case class MLRFClassificationModel(model: RandomForestClassificationModel) extends TreeBasedModel
case class MLGBTRegressionModel(model: GBTRegressionModel) extends TreeBasedModel
case class MLGBTClassificationModel(model: GBTClassificationModel) extends TreeBasedModel

/**
 * Parent class for DecisionTree-based tests which run on a large dataset.
 */
abstract class DecisionTreeTests(sc: SparkContext)
  extends RegressionAndClassificationTests[TreeBasedModel](sc) {

  val TEST_DATA_FRACTION =
    ("test-data-fraction",  "fraction of data to hold out for testing (ignored if given training and test dataset)")
  val LABEL_TYPE =
    ("label-type", "Type of label: 0 indicates regression, 2+ indicates " +
      "classification with this many classes")
  val FRAC_CATEGORICAL_FEATURES = ("frac-categorical-features",
    "Fraction of features which are categorical")
  val FRAC_BINARY_FEATURES =
    ("frac-binary-features", "Fraction of categorical features which are binary. " +
      "Others have 20 categories.")
  val TREE_DEPTH = ("tree-depth", "Depth of true decision tree model used to label examples.")
  val MAX_BINS = ("max-bins", "Maximum number of bins for the decision tree learning algorithm.")
  val NUM_TREES = ("num-trees", "Number of trees to train.  If 1, run DecisionTree.  If >1, run an ensemble method (RandomForest).")
  val FEATURE_SUBSET_STRATEGY =
    ("feature-subset-strategy", "Strategy for feature subset sampling. Supported: auto, all, sqrt, log2, onethird.")
  val ALG_TYPE = ("alg-type", "Algorithm: byRow for original MLlib, byCol for Yggdrasil")

  intOptions = intOptions ++ Seq(LABEL_TYPE, TREE_DEPTH, MAX_BINS, NUM_TREES)
  doubleOptions = doubleOptions ++ Seq(TEST_DATA_FRACTION, FRAC_CATEGORICAL_FEATURES, FRAC_BINARY_FEATURES)
  stringOptions = stringOptions ++ Seq(FEATURE_SUBSET_STRATEGY, ALG_TYPE)

  addOptionalOptionToParser("training-data", "path to training dataset (if not given, use random data)", "", classOf[String])
  addOptionalOptionToParser("test-data", "path to test dataset (only used if training dataset given)" +
      " (if not given, hold out part of training data for validation)", "", classOf[String])

  var categoricalFeaturesInfo: Map[Int, Int] = Map.empty

  protected var labelType = -1

  def runTest(
    rdd: DataFrame, transposedDataset: Option[(RDD[(Int, Vector)], Array[Double])]): TreeBasedModel

  override def run(): JValue = {
    val algType: String = stringOptionValue(ALG_TYPE)
    val labelType: Int = intOptionValue(LABEL_TYPE)
    val trainDF = DataGenerator.setMetadata(rdd, categoricalFeaturesInfo, labelType)
    // Transpose dataset before timing "byCol"
    var transposeTime: Double = 0.0
    val transposedDataset = algType match {
      case "byRow" => None
      case "byCol" => {
        val start = System.currentTimeMillis()
        val dt = new DecisionTreeRegressor()
          .setFeaturesCol("features")
          .setLabelCol("label")
          .setMaxDepth(10)
          .setAlgorithm("byCol")
        val (columns, labels) = dt.transposeDataset(trainDF)
        columns.persist(StorageLevel.MEMORY_AND_DISK)
        columns.count()
        transposeTime = (System.currentTimeMillis() - start).toDouble / 1001.0
        println(s"transposeTime: $transposeTime")
        sc.parallelize(0 until 256, 256).foreach(x => System.gc())
        Some((columns, labels))
      }
      case _ => throw new IllegalArgumentException(s"Got unknown algType: $algType")
    }

    // var start = System.currentTimeMillis()
    // val model = runTest(trainDF, transposedDataset)
    // val trainingTime = (System.currentTimeMillis() - start).toDouble / 1000.0

    // start = System.currentTimeMillis()
    // val trainingMetric = validate(model, rdd)
    // val testTime = (System.currentTimeMillis() - start).toDouble / 1000.0

    // val testMetric = validate(model, testRdd)
    // println(s"trainingTime: $trainingTime")
    // println(s"trainingMetric: $trainingMetric")
    // println(s"testTime: $testTime")
    // println(s"testMetric: $testMetric")
    Map("trainingTime" -> transposeTime, "testTime" -> transposeTime,
      "trainingMetric" -> transposeTime, "testMetric" -> transposeTime)
  }


  def validate(model: TreeBasedModel, rdd: RDD[LabeledPoint]): Double = {
    val numExamples = rdd.count()
    val predictions: RDD[(Double, Double)] = model match {
      case MLDTRegressionModel(rfModel) => makePredictions(rfModel, rdd)
      case MLDTClassificationModel(rfModel) => makePredictions(rfModel, rdd)
      case MLRFRegressionModel(rfModel) => makePredictions(rfModel, rdd)
      case MLRFClassificationModel(rfModel) => makePredictions(rfModel, rdd)
      case MLGBTRegressionModel(gbtModel) => makePredictions(gbtModel, rdd)
      case MLGBTClassificationModel(gbtModel) => makePredictions(gbtModel, rdd)
      case _ =>
        throw new Exception(s"Unknown match error.  Got type: ${model.getClass.getName}")
    }
    val labelType: Int = intOptionValue(LABEL_TYPE)
    if (labelType == 0) {
      calculateRMSE(predictions, numExamples)
    } else {
      calculateAccuracy(predictions, numExamples)
    }
  }

  // TODO: generate DataFrame outside of `runTest` so it is not included in timing results
  private def makePredictions(
      model: PredictionModel[Vector, _], rdd: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    val labelType: Int = intOptionValue(LABEL_TYPE)
    val dataFrame = DataGenerator.setMetadata(rdd, categoricalFeaturesInfo, labelType)
    val results = model.transform(dataFrame)
    results
      .select(model.getPredictionCol, model.getLabelCol)
      .map { case Row(prediction: Double, label: Double) => (prediction, label) }
  }
}

class DecisionTreeTest(sc: SparkContext) extends DecisionTreeTests(sc) {
  val supportedTreeTypes = Array("DecisionTree") // "RandomForest", "GradientBoostedTrees"

  val ENSEMBLE_TYPE =
    ("ensemble-type", "Type of ensemble algorithm: " + supportedTreeTypes.mkString(" "))

  stringOptions = stringOptions ++ Seq(ENSEMBLE_TYPE)

  val options = intOptions ++ stringOptions ++ booleanOptions ++ doubleOptions ++ longOptions
  addOptionsToParser()

  private def getTestDataFraction: Double = {
    val testDataFraction: Double = doubleOptionValue(TEST_DATA_FRACTION)
    assert(testDataFraction >= 0 && testDataFraction <= 1,
      s"Bad testDataFraction: $testDataFraction")
    testDataFraction
  }

  override def createInputData(seed: Long) = {
    val trainingDataPath: String = optionValue[String]("training-data")
    val (rdds, categoricalFeaturesInfo_, numClasses) = if (trainingDataPath != "") {
      println(s"LOADING FILE: $trainingDataPath")
      val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
      val testDataPath: String = optionValue[String]("test-data")
      val testDataFraction: Double = getTestDataFraction
      DataLoader.loadLibSVMFiles(sc, numPartitions, trainingDataPath, testDataPath,
        testDataFraction, seed)
    } else {
      createSyntheticInputData(seed)
    }
    assert(rdds.length == 2)
    rdd = rdds(0).cache()
    testRdd = rdds(1)
    categoricalFeaturesInfo = categoricalFeaturesInfo_
    this.labelType = numClasses

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }

  /**
   * Create synthetic training and test datasets.
   * @return (trainTestDatasets, categoricalFeaturesInfo, numClasses) where
   *          trainTestDatasets = Array(trainingData, testData),
   *          categoricalFeaturesInfo is a map of categorical feature arities, and
   *          numClasses = number of classes label can take.
   */
  private def createSyntheticInputData(
      seed: Long): (Array[RDD[LabeledPoint]], Map[Int, Int], Int) = {
    // Generic test options
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    // Data dimensions and type
    val numExamples: Long = longOptionValue(NUM_EXAMPLES)
    val numFeatures: Int = intOptionValue(NUM_FEATURES)
    val labelType: Int = intOptionValue(LABEL_TYPE)
    val fracCategoricalFeatures: Double = doubleOptionValue(FRAC_CATEGORICAL_FEATURES)
    val fracBinaryFeatures: Double = doubleOptionValue(FRAC_BINARY_FEATURES)
    // Model specification
    val treeDepth: Int = intOptionValue(TREE_DEPTH)

    // val rdd_ = DataGenerator.generateFriedman1LabeledPoints(sc, math.ceil(numExamples * 1.25).toLong,
    // numFeatures, 0, numPartitions, seed)
    val (rdd_, categoricalFeaturesInfo_) =
      DataGenerator.generateDecisionTreeLabeledPoints(sc, math.ceil(numExamples * 1.25).toLong,
        numFeatures, numPartitions, labelType,
        fracCategoricalFeatures, fracBinaryFeatures, treeDepth, seed)
    // MLUtils.saveAsLibSVMFile(rdd_, "hdfs://ec2-54-85-108-155.compute-1.amazonaws.com:9000/temp/yahoo")
    categoricalFeaturesInfo = categoricalFeaturesInfo_

    val splits = rdd_.randomSplit(Array(0.8, 0.2), seed)
    (splits, categoricalFeaturesInfo, labelType)
  }

  // Count dataset transposition time as part of training by default
  override def runTest(rdd: DataFrame): TreeBasedModel = runTest(rdd, None)

  // Will use precomputed `transposedDataset` if available
  override def runTest(
      dataset: DataFrame, transposedDataset: Option[(RDD[(Int, Vector)], Array[Double])]): TreeBasedModel = {
    val treeDepth: Int = intOptionValue(TREE_DEPTH)
    val maxBins: Int = intOptionValue(MAX_BINS)
    val numTrees: Int = intOptionValue(NUM_TREES)
    val featureSubsetStrategy: String = stringOptionValue(FEATURE_SUBSET_STRATEGY)
    val ensembleType: String = stringOptionValue(ENSEMBLE_TYPE)
    val algType: String = stringOptionValue(ALG_TYPE)
    if (!supportedTreeTypes.contains(ensembleType)) {
      throw new IllegalArgumentException(
        s"DecisionTreeTest given unknown ensembleType param: $ensembleType." +
        " Supported values: " + supportedTreeTypes.mkString(" "))
    }
    if (labelType == 0) {
      // Regression
      ensembleType match {
        case "DecisionTree" =>
          val dtRegressor = new DecisionTreeRegressor()
            .setImpurity("variance")
            .setMaxDepth(treeDepth)
            .setMaxBins(maxBins)
            .setAlgorithm(algType)
          val model = transposedDataset match {
            case None => dtRegressor.fit(dataset)
            case Some((columns, labels)) => dtRegressor.fit(dataset, columns, labels)
          }
          println(s"Tree depth: ${model.depth}")
          if (model.numNodes < 200) {
            println(model.toDebugString) // Print full model.
          } else {
            println(model) // Print model summary.
          }
          MLDTRegressionModel(model)
          /*
        case "RandomForest" =>
          val model = new RandomForestRegressor()
            .setImpurity("variance")
            .setMaxDepth(treeDepth)
            .setMaxBins(maxBins)
            .setAlgorithm(algType)
            .setNumTrees(numTrees)
            .setFeatureSubsetStrategy(featureSubsetStrategy)
            .setSeed(this.getRandomSeed)
            .fit(dataset)
          MLRFRegressionModel(model)
        case "GradientBoostedTrees" =>
          val model = new GBTRegressor()
            .setLossType("squared")
            .setMaxBins(maxBins)
            .setAlgorithm(algType)
            .setMaxDepth(treeDepth)
            .setMaxIter(numTrees)
            .setStepSize(0.1)
            .setSeed(this.getRandomSeed)
            .fit(dataset)
          MLGBTRegressionModel(model)
          */
      }
    } else if (labelType >= 2) {
      // Classification
      ensembleType match {
        case "DecisionTree" =>
          val dtClassifier = new DecisionTreeClassifier()
            .setImpurity("gini")
            .setMaxDepth(treeDepth)
            .setMaxBins(maxBins)
            .setAlgorithm(algType)
          val model = transposedDataset match {
            case None => dtClassifier.fit(dataset)
            case Some((columns, labels)) => dtClassifier.fit(dataset, columns, labels)
          }
          println(s"Tree depth: ${model.depth}")
          if (model.numNodes < 200) {
            println(model.toDebugString) // Print full model.
          } else {
            println(model) // Print model summary.
          }
          MLDTClassificationModel(model)
          /*
        case "RandomForest" =>
          val model = new RandomForestClassifier()
            .setImpurity("gini")
            .setMaxDepth(treeDepth)
            .setMaxBins(maxBins)
            .setAlgorithm(algType)
            .setNumTrees(numTrees)
            .setFeatureSubsetStrategy(featureSubsetStrategy)
            .setSeed(this.getRandomSeed)
            .fit(dataset)
          MLRFClassificationModel(model)
        case "GradientBoostedTrees" =>
          val model = new GBTClassifier()
            .setLossType("logistic")
            .setMaxBins(maxBins)
            .setAlgorithm(algType)
            .setMaxDepth(treeDepth)
            .setMaxIter(numTrees)
            .setStepSize(0.1)
            .setSeed(this.getRandomSeed)
            .fit(dataset)
          MLGBTClassificationModel(model)
          */
      }
    } else {
      throw new IllegalArgumentException(s"Bad label-type parameter " +
        s"given to DecisionTreeTest: $labelType")
    }
  }
}
