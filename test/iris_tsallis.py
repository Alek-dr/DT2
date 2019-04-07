import pandas as pd
import graphviz
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot
from core.graph import Node
from numpy import zeros
from utils.help_functions import train_test_split

iris = pd.read_csv('../datasets/Iris.csv')
iris.drop("Id",axis=1,inplace=True)

trainData, testData = train_test_split(iris)

def train():
    dt = DecisionTree()
    dt.learn(data=trainData, target='Species', criterion='Tsallis', q=5)
    out_name = 'tennis_ts'
    dot_data = export2dot(out_name, dt.tree, writeId=True)

if __name__ == '__main__':
    train()