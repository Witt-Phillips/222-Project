import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = 'AI On-Campus Research Survey (Responses).xlsx'
df = pd.read_excel(path, engine='openpyxl')

print(df.head())

# plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
# plt.show()

if __name__ == '__main__':
    print("Hello world!")

    # Read input
    col_input = input("Enter prediction cols by #, separated by space:")
    active_cols = [df.columns[int(num)] for num in col_input.split()]
        #print(active_cols)
    
    