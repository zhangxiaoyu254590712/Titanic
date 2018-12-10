import numpy as np
import pandas as pd
from data_processing import data_proc

"""
file = './raw_data/train.csv'
df = pd.read_csv(file, header=0)
df['Imputed'] = 0
a=np.isnan(df['Age'])
nan_index = df.loc[a].index
df.loc[nan_index,'Age'] = df.loc[nan_index,'']

print(df.loc[nan_index,'Age'])
"""
a = data_proc('./raw_data/train.csv','./raw_data/test.csv')
train_df,_ = a.total_data_proc(5)
print(train_df.loc[:,train_df.columns!='Survived'].head())
print(train_df[train_df.loc[:,train_df.columns!='Survived'].columns].head())

