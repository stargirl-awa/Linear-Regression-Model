import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 加载数据
df_train = pd.read_csv('train.csv')

# 重点关注字段
cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF',
        'GarageCars', 'FullBath', 'YearBuilt']

# 处理丢失数据
missing_data = df_train.isnull().sum().sort_values(ascending=False)
df_train = df_train.drop((missing_data[missing_data > 1]).index, axis=1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

# 处理离群值
df_train = df_train[df_train['GrLivArea'] < 4500]
