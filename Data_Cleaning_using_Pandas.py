#!/usr/bin/env python
# coding: utf-8

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
from IPython.core.getipython import get_ipython
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# remove warnings
import warnings
warnings.filterwarnings('ignore')

#Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print (train_data.shape)

#Remove columns
previous_num_columns = train_data.select_dtypes(exclude=['object']).columns.values.tolist()
previous_num_columns.remove('Id')
previous_num_columns.remove('SalePrice')

#Drop columns
train_data.drop(train_data[train_data["LotFrontage"] > 200].index, inplace=True)
train_data.drop(train_data[train_data["LotArea"] > 70000].index, inplace=True)
train_data.drop(train_data[train_data["MasVnrArea"] > 1500].index, inplace=True)


train_length = train_data.shape[0]

def fill_missing_conbined_data(column, value):
    conbined_data.loc[conbined_data[column].isnull(),column] = value

#Concat data
conbined_data = pd.concat([train_data.loc[:, : 'SalePrice'], test_data])
conbined_data = conbined_data[test_data.columns]
display(conbined_data.head(1))

#Group by and plot
conbined_data['LotFrontage'].groupby(conbined_data["Neighborhood"]).median().plot()
conbined_data['LotFrontage'].groupby(conbined_data["Neighborhood"]).mean().plot()

#isnull 

lf_neighbor_map = conbined_data['LotFrontage'].groupby(conbined_data["Neighborhood"]).median()
    
rows = conbined_data['LotFrontage'].isnull()
conbined_data['LotFrontage'][rows] = conbined_data['Neighborhood'][rows].map(lambda neighbor : lf_neighbor_map[neighbor])

conbined_data[conbined_data['LotFrontage'].isnull()]

#fill missin data
fill_missing_conbined_data('Alley', 'NA')

#fillna
conbined_data['MasVnrType'].fillna('None', inplace=True)
conbined_data['MasVnrArea'].fillna(conbined_data['MasVnrArea'].median(), inplace=True)

#plot
sns.countplot(conbined_data['Electrical'])

#save to csv
conbined_data.to_csv("data/conbined_data.csv", index=False)
