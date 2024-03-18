import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#Read & drop unused columns
df = pd.read_csv('data.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
og_df = df
#print(df.head())

def preprocess(data):
    # Normalize continuous variables (if they exist)
    to_normalize = ['Age', 'SibSp', 'Parch', 'Fare', 'PClass']
    to_normalize = [c for c in to_normalize if c in data.columns]
    if to_normalize:
        data[to_normalize] = MinMaxScaler().fit_transform(data[to_normalize])  

    #woman = 0, man = 1
    if 'Sex' in data.columns:
        pd.set_option('future.no_silent_downcasting', True)
        data['Sex'] = data['Sex'].replace({'female': 0, 'male': 1})
    
    data = data.dropna(axis=0, how='any')
    #print(data.head())
    return data

def lsr(data, predict):
    features = data.drop('Survived', axis=1)
    target = data['Survived']
    
    #split training/ testing sets 75/25
    f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.25, random_state=1)

    #print(f_train.head())
    
    #Fit & evaluate model
    m = LinearRegression()
    m.fit(f_train, t_train)
    pred = m.predict(f_test)    
    ms = mean_squared_error(t_test, pred)
    r2 = r2_score(t_test, pred)

    print("\nEntering summary stats module...")
    print("Model trained on:")
    for col in features.columns:
        print(f"*{col}")
    print(f"Mean Squared Error: {round(ms, 2)}\nr2 score: {round(r2, 2)}")
    
    if predict:
        print("\nEntering prediction module...")
        print("When prompted, provide information on an individual to see their odds of survival on the Titanic.")
        
        inputs = {}
        for col in features.columns:
            inp = input(f"Enter {col}: ")
            inputs[col] = [inp]

        inp_df = pd.DataFrame(inputs)

        output = m.predict(inp_df)
        print(f"Predicted survival value: {round(output[0], 2)}")

if __name__ == '__main__':
    # Read input && create active dataframe (adf) from selected cols
    col_input = input("Enter prediction cols by #, separated by space: ")
    idxs = [int(num) for num in col_input.split()]
    active_cols = [df.columns[0]] + [df.columns[idx] for idx in idxs]
    df = df.loc[:, active_cols]
    #print(adf.head())

    #if false, don't give prediction option
    df = preprocess(df)
    lsr(df, True)

    #print(df.head())
