import pandas as pd
import graphviz
from core.decision_tree import DecisionTree
from graph_visualize.dot_convertor import export2dot
from utils.help_functions import train_test_split

titanic = pd.read_csv('../../datasets/titanic/titanic.csv')
titanic.drop(['Name'], axis=1, inplace=True)
trainData, testData = train_test_split(titanic)

categorical = ('Survived','Pclass','Ticket')

def train():
    tsall = DecisionTree()
    tsall.learn(data=trainData, target='Survived', criterion='Tsallis',
                  as_categorial=('Survived', 'Pclass', 'Ticket'), q=1.5)
    # tsall.C45(data=trainData, target='Survived', as_categorial=('Survived', 'Pclass', 'Ticket'))
if __name__=='__main__':
    train()