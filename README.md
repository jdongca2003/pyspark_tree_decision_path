pyspark tree decision paths
--
This library extracts pyspark decision tree paths for a given input.

Introduction
--
Spark decision tree models such as random forest and gbdt classifiers are implemented in Scala. No APIs are provided to extract decision paths for a given input.
In some applications, explicit decision tree paths are useful for non-expert users to understand how decisions are made. 

In this repository, we provide a simple utility to convert spark decision tree (pyspark.ml.classification.DecisionTreeClassificationModel or pyspark.ml.regression.DecisionTreeRegressionModel) in scala to tree in python and extract decision paths for a given input. It is also useful if you like to port spark tree model into non-spark environment.

This library can be used to extract decision paths for pyspark.ml.classification.{RandomForestClassificationModel, GBTClassificationModel} and pyspark.ml.regression.{RandomForestRegressionModel, GBTRegressionModel}.

Get-started
--
    1. Create wheel package
       python setup.py bdist_wheel

    2. Upload the wheel package into databrick cluster

    3. Run the example notebook in databricks

      ```
      from pyspark_tree_decision_paths import tree, decision_path
      python_tree_model = tree.build_tree(spark_tree_model)
      predicted_paths = decision_path.extract_decision_tree_paths(python_tree_model, x)
      print_decision_paths(predicted_paths)

      #decison path outputs look like for binary classification ( <=50K, >50K) on [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)

      feature_name: relationship_index feature_value:  Husband split_value: [" Not-in-family", " Own-child", " Unmarried", " Other-relative"] split: not in
      feature_name: education-num feature_value: 10.0 split_value: 12.5 split: <=
      feature_name: capital-gain feature_value: 7688.0 split_value: 6808.0 split: >
      Leaf prediction:  >50K

      ```


