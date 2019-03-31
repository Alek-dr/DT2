import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('../datasets/Iris.csv')
sns.pairplot(data, hue='Species')
plt.show()