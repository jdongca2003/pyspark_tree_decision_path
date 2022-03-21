# Databricks notebook source
# MAGIC %pip install pyspark_tree_decision_paths-0.0.1-py3-none-any.whl

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

#download data from https://archive.ics.uci.edu/ml/datasets/adult and upload them
#train: adult.data
#test: adult.test
train_data_path = YOUR_DATA_PATH for adult.data
test_data_path = YOUR_DATA_PATH for adult.test

# COMMAND ----------

import pandas as pd
cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
train_data = pd.read_csv(train_data_path, names=cols)
train_data.head(10)

# COMMAND ----------

train_data.nunique()

# COMMAND ----------

test_data = pd.read_csv(test_data_path, names=cols)
test_data.head(10)

# COMMAND ----------

train_df = spark.createDataFrame(train_data)
test_df = spark.createDataFrame(test_data)
train_df.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F
test_df = test_df.withColumn("label", F.regexp_replace("label", '\.', ''))
display(test_df)

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="label", outputCol="target", handleInvalid='keep')
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
continuous_cols =["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

# COMMAND ----------

stages=[]
categorical_index = ['{}_index'.format(e) for e in categorical_cols]
indexer = StringIndexer(inputCols=categorical_cols, outputCols=categorical_index, handleInvalid='keep')

# COMMAND ----------

from pyspark.ml.feature import StringIndexer,VectorAssembler
final_assembler = VectorAssembler(inputCols=categorical_index + continuous_cols, outputCol='feature_vec')

# COMMAND ----------

from pyspark.ml import Pipeline
stages = [labelIndexer, indexer, final_assembler]

# COMMAND ----------

train_df = train_df.fillna(0, subset=continuous_cols)
train_df = train_df.fillna("unknown", subset=categorical_cols)
test_df = test_df.fillna(0, subset=continuous_cols)
test_df = test_df.fillna("unknown", subset=categorical_cols)

# COMMAND ----------

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(train_df)
new_train_df = pipelineModel.transform(train_df)
new_test_df = pipelineModel.transform(test_df)

# COMMAND ----------

dt = DecisionTreeClassifier(labelCol="target", featuresCol="feature_vec", maxBins=48)
tree_model = dt.fit(new_train_df)

# COMMAND ----------

display(new_test_df)

# COMMAND ----------

predictions=tree_model.transform(new_test_df)

# COMMAND ----------

display(predictions)

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(
             labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g " % accuracy)

# COMMAND ----------

print(type(tree_model))

# COMMAND ----------

from pyspark_tree_decision_paths import tree, decision_path

# COMMAND ----------

my_tree_model = tree.build_tree(tree_model)

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
new_pred = predictions.withColumn("features_array", vector_to_array("feature_vec")).limit(10)
pred_pandas = new_pred.toPandas()
display(pred_pandas)

# COMMAND ----------

x = pred_pandas["features_array"].iloc[3]

# COMMAND ----------

print(decision_path.tree_predict(my_tree_model, x))

# COMMAND ----------

predicted_paths = decision_path.extract_decision_tree_paths(my_tree_model, x)

# COMMAND ----------

name_index = new_train_df.schema["feature_vec"].metadata["ml_attr"]["attrs"]
index_to_feature_names={}
for e in name_index['numeric']:
   idx = e['idx']
   feature_name = e['name']
   index_to_feature_names[idx] = feature_name

categorical_feature_values = {}
for e in name_index['nominal']:
   idx = e['idx']
   feature_name = e['name']
   index_to_feature_names[idx] = feature_name
   categorical_feature_values[feature_name] = e['vals']

# COMMAND ----------

targetIndex_to_name = pipelineModel.stages[0].labels
print(targetIndex_to_name)

# COMMAND ----------

#map categorical_index to category name
category_original_name_map = {'{}_index'.format(e) : e for e in categorical_cols}
print(category_original_name_map)

# COMMAND ----------

import json

def print_node(d_node):
    split_branch = d_node.branch
    feature_val = d_node.feature_value
    node = d_node.node
    if node.is_categorical_split:
       feature_index = node.feature_index
       feature_name = index_to_feature_names[feature_index]
       category_values = categorical_feature_values[feature_name]
       feature_val_cat_name = category_values[int(feature_val)] 
       values = [category_values[int(idx)] for idx in sorted(list(node.split_value))[:4] ]
       if split_branch == decision_path.TreeBranch.left:
           split_msg = "in"
       else:
           split_msg = "not in" 
       left_category_set_str = json.dumps(values)
       print(f'Feature {category_original_name_map[feature_name]}({feature_val_cat_name}) is {split_msg} {left_category_set_str}')
    elif node.feature_index != None:
       if split_branch == decision_path.TreeBranch.left:
          split_msg = "<="
       else:
          split_msg = ">"
       feature_name = index_to_feature_names[node.feature_index]
       print(f'Feature {feature_name}({feature_val}) {split_msg} {node.split_value}')
    else:
       class_idx = int(node.prediction)
       pred_class_name = targetIndex_to_name[class_idx]
       print(f'Leaf prediction: {pred_class_name}')


def print_decision_paths(tree_decision_path):
    for d_node in tree_decision_path:
        print_node(d_node)

# COMMAND ----------

print_decision_paths(predicted_paths)
