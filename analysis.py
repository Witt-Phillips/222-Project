import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('data.csv')
og_df = df
#print(df.head())

def preprocess(data):
    #Drop currently unused columns
    data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    # Normalize continuous variables (if they exist)
    to_normalize = ['Age', 'SibSp', 'Parch', 'Fare', 'PClass']
    to_normalize = [c for c in to_normalize if c in data.columns]
    if to_normalize:
        data[to_normalize] = MinMaxScaler().fit_transform(data[to_normalize])  
    #print(data[to_normalize].head())

    #woman = 0, man = 1
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].replace({'female': 0, 'male': 1})

if __name__ == '__main__':
    print("Hello world!")

    preprocess(df)

    # Read input && create active dataframe (adf) from selected cols
    col_input = input("Enter prediction cols by #, separated by space: ")
    idxs = [int(num) for num in col_input.split()]
    active_cols = [df.columns[idx] for idx in idxs]
    adf = df.loc[:, active_cols]
    #print(adf.head())


    
