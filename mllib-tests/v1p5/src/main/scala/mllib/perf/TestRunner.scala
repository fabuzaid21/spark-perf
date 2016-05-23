package mllib.perf

import scala.collection.JavaConverters._

import org.json4s.JsonAST._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import ml.dmlc.xgboost4j.scala.Booster

import org.apache.spark.{SparkConf, SparkContext}


object TestRunner {
    def main(args: Array[String]) {
      if (args.length < 1) {
        println(
          "mllib.perf.TestRunner requires 1 or more args, you gave %s, exiting".format(args.length))
        System.exit(1)
      }
      val testName = args(0)
      val perfTestArgs = args.slice(1, args.length)
      val sc = new SparkContext(new SparkConf().setAppName("TestRunner: " + testName))
      sc.registerKryoClasses(Array(classOf[Booster]))


      // Unfortunate copy of code because there are Perf Tests in both projects and the compiler doesn't like it
      val test: PerfTest = testName match {
        // trees
        case "decision-tree" => new DecisionTreeTest(sc)
      }
      test.initialize(testName, perfTestArgs)
      // Generate a new dataset for each test
      val rand = new java.util.Random(test.getRandomSeed)

      val numTrials = test.getNumTrials
      val interTrialWait = test.getWait

      var testOptions: JValue = test.getOptions
      val results: Seq[JValue] = (1 to numTrials).map { i =>
        test.createInputData(rand.nextLong())
        val res: JValue = test.run()
        System.gc()
        sc.parallelize(0 until 256, 256).foreach(x => System.gc())
        Thread.sleep(interTrialWait)
        res
      }
      // Report the test results as a JSON object describing the test options, Spark
      // configuration, Java system properties, as well as the per-test results.
      // This extra information helps to ensure reproducibility and makes automatic analysis easier.
      val json: JValue =
        ("testName" -> testName) ~
        ("options" -> testOptions) ~
        ("sparkConf" -> sc.getConf.getAll.toMap) ~
        ("sparkVersion" -> sc.version) ~
        ("systemProperties" -> System.getProperties.asScala.toMap) ~
        ("results" -> results)
      println("results: " + compact(render(json)))

      sc.stop()
  }
}
