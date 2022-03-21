from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.regression import DecisionTreeRegressionModel

class TreeNode:
    def __init__(self,
                  feature_index,
                  is_categorical_split=False,
                  split_value=None,
                  prediction=None,
                  left_child=None,
                  right_child=None):
        #for leaf node, feature_index is None
        self.feature_index = feature_index
        self.left  = left_child
        self.right = right_child
        self.is_categorical_split = is_categorical_split
        #when is_categorical_split is True, split_value is set otherwise a threshold
        self.split_value = split_value
        #prediction is only available for leaf node
        self.prediction = prediction

def build_tree(tree_model: (DecisionTreeClassificationModel, DecisionTreeRegressionModel)) -> TreeNode:
    """ build a tree from pyspark.ml.classification.DecisionTreeClassificationModel or DecisionTreeRegressionModel

    Return:
        tree root node
    """
    def build_tree_(spark_tree_node) -> TreeNode:
        if spark_tree_node.numDescendants() == 0:
            prediction = spark_tree_node.prediction()
            node = TreeNode(None, prediction=prediction)
            return node
        #internal node
        feature_index = spark_tree_node.split().featureIndex()
        if str(spark_tree_node.split().getClass()).endswith("tree.CategoricalSplit"):
            vals = list(spark_tree_node.split().leftCategories())
            split_val = set(vals)
            is_categorical_split = True
        else:
            split_val = spark_tree_node.split().threshold()
            is_categorical_split = False

        left_child = build_tree_(spark_tree_node.leftChild())
        right_child = build_tree_(spark_tree_node.rightChild())
        node = TreeNode(feature_index,
                        is_categorical_split=is_categorical_split,
                        split_value=split_val,
                        left_child=left_child,
                        right_child=right_child)
        return node
    if isinstance(tree_model, (DecisionTreeClassificationModel, DecisionTreeRegressionModel)):
        return build_tree_(tree_model._call_java('rootNode'))
    else:
        raise ValueError(f"tree_model: {type(tree_model)} is neither DecisionTreeClassificationModel nor DecisionTreeRegressionModel)")

