import pandas as pd
from core.decision_tree import DecisionTree

tennis = pd.read_csv('../datasets/PlayTennis_0.csv')
tennis.drop("Day",axis=1,inplace=True)

dt = DecisionTree()
dt.ID3(data=tennis, target='Play')
print()
