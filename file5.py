import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv('/VSCODE/heart.csv')
X = df.drop('target', axis=1)
y = df['target']
df.head()