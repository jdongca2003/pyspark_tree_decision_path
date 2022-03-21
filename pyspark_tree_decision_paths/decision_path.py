import enum
from typing import List
from .tree import TreeNode

class TreeBranch(enum.Enum):
      left = 0
      right = 1
      leaf = 2

class DecisionNode:
    def __init__ (self, branch: TreeBranch, feature_value, node: TreeNode):
        self.branch = branch
        self.feature_value = feature_value
        self.node = node


def shouldGoLeftAtSplit(node, val):
    if node.is_categorical_split:
          return val in node.split_value
    return val <= node.split_value


def extract_decision_tree_paths(tree_root: TreeNode, x) -> List[DecisionNode]:
    """ Extract decision paths for a given input feature vector x

    Arguments:
        tree_root --- decision tree
        x --- list or numpy array
    Return:
        a list of DecisionNode
    """
    decisions = []
    node = tree_root
    while node.feature_index != None:
        feature_val = x[node.feature_index]
        if shouldGoLeftAtSplit(node, feature_val):
            decisions.append(DecisionNode(TreeBranch.left, feature_val, node))
            node = node.left
        else:
            decisions.append(DecisionNode(TreeBranch.right, feature_val, node))
            node = node.right
    decisions.append(DecisionNode(TreeBranch.leaf, None, node))
    return decisions


def tree_predict(tree_root: TreeNode, x):
    """ extract decision value in the leaf node for a given input x

    Arguments:
        tree_root -- decision tree
        x  -- list or numpy array

    Return:
        prediction
    """
    node = tree_root
    while node.feature_index != None:
        feature_val = x[node.feature_index]
        if shouldGoLeftAtSplit(node, feature_val):
            node = node.left
        else:
            node = node.right
    return node.prediction

