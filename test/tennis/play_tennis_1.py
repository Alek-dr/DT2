import pandas as pd
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot

tennis = pd.read_csv('../datasets/PlayTennis_1.csv')
tennis.drop("Day",axis=1,inplace=True)

dt = DecisionTree()
dt.C45(data=tennis, target='Play')


# zenit = pd.read_csv('../datasets/Zenit.csv')
#
# dt = DecisionTree()
# dt.C45(data=zenit, target='Win')

# out_name = 'tennis_0'
# export2dot(out_name, dt.tree)
