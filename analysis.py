import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.figure_factory as ff

#Read & drop unused columns
df = pd.read_csv('data.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
og_df = df

vars = {
        "Class (1-3)": "Pclass",
        "Age (0-80)": "Age",
        "\# of Siblings/Spouse Onboard (0-8)": "SibSp",
        "\# of Parents/Children Onboard (0-6)": "Parch",
        "Fare ($0-512)": "Fare"
    }

def select_features():
    if 'active_cols' not in st.session_state:
        st.session_state['active_cols'] = ['Survived']

    for human_var, df_var in vars.items():
        if st.checkbox(human_var, key=df_var):
            if df_var not in st.session_state['active_cols']:
                st.session_state['active_cols'].append(df_var)
        else:
            if df_var in st.session_state['active_cols']:
                st.session_state['active_cols'].remove(df_var)
    
    return st.session_state['active_cols']

def preprocess(data):
    # # Normalize continuous variables (if they exist)
    # to_normalize = ['Age', 'SibSp', 'Parch', 'Fare', 'PClass']
    # to_normalize = [c for c in to_normalize if c in data.columns]
    # if to_normalize:
    #     data[to_normalize] = MinMaxScaler().fit_transform(data[to_normalize])  

    #woman = 0, man = 1
    if 'Sex' in data.columns:
        pd.set_option('future.no_silent_downcasting', True)
        data['Sex'] = data['Sex'].replace({'female': 0, 'male': 1})
    
    data = data.dropna(axis=0, how='any')
    return data

def corr_plot(data):
    corr = data.corr()
   
    figure = ff.create_annotated_heatmap(
         z = corr.to_numpy(),
        x = corr.columns.to_list(),
        y = corr.columns.to_list(),
        annotation_text=corr.round(2).to_numpy(),
        colorscale='Magma',
        showscale=True
    )
    figure.update_layout(
        title_text = 'Correlation Heatmap',
        title_x = 0.4
    )
    return figure

def lsr(data, predict):
    features = data.drop('Survived', axis=1)
    target = data['Survived']
    
    #split training/ testing sets 75/25
    f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.25, random_state=1)
    
    #Fit & evaluate model
    m = LinearRegression()
    m.fit(f_train, t_train)
    pred = m.predict(f_test)    
    ms = mean_squared_error(t_test, pred)
    r2 = r2_score(t_test, pred)

    # st.write('Features considered:')
    # for colname in features.columns:
    #     st.write(colname)
    #st.write(f"Mean Squared Error: {round(ms, 2)}\nr2 score: {round(r2, 2)}")

    st.markdown(f"""
        <p style='font-size: 25px;'>
            Mean Squared Error: <strong>{round(ms, 2)}</strong><br>
            r<sup>2</sup> score: <strong>{round(r2, 2)}</strong>
        </p>
        """, unsafe_allow_html=True)

#TODO: explain how we calculate/ interpret these values w/ linear algebra

#TODO: on the right, we could plot correlations or the lsr

    #Survival Score Prediction
    st.header('Survival Score by Features')
    st.write("Provide information on an individual to see their odds of survival on the Titanic. Ranges from the dataset are provided for context, but you're welcome to explore parameters outside of these bounds.")
    
    if 'inputs' not in st.session_state:
        st.session_state['inputs'] = {}

    inputs = {}
    inv_vars = {val: key for key, val in vars.items()}

    for col in features.columns:
        name = inv_vars.get(col)
        st.session_state['inputs'][col] = st.number_input(f"Enter {name}: ", key=f"input_{col}", value=st.session_state['inputs'].get(col, 0))

    if st.button('Predict'):
        inp_df = pd.DataFrame([st.session_state['inputs']])
        output = m.predict(inp_df)
        #st.write(f"Predicted survival value: {round(output[0], 2)}")
        st.markdown(f"""
        <p style='font-size: 25px;'>
            Predicted survival value: <strong>{round(output[0], 2)}</strong><br>
        </p>
        """, unsafe_allow_html=True)
    
#TODO: interpret this prediction & explain what it means
        
#TODO: maybe a little exploratory data section? Like do some examples and show how we can use
#lsr to find variables that are predictive of survival?

if __name__ == '__main__':
    #Setup app
    st.title('The Unsinkable Ship: Who survived the Titanic?')
    st.image("titanic_drawing.jpeg", caption="Titanic Sinking, Willy Stöwer. Wikimedia Commons.", use_column_width=True)
    st.write('The [Titanic](https://www.history.com/topics/early-20th-century-us/titanic#unsinkable-titanic-s-fatal-flaws), deemed "practically unsinkable" by experts, sunk in 1912. More than 1,500 of the 2,240 passengers onboard were lost. Here, we use passenger data to explore the factors that contributed to surival. Select features to see how well they predict survial via least squares regression.')
    
    # Select columns
    st.header('Select features')            
    active_cols = select_features()

    if 'lsr_button' not in st.session_state:
        st.session_state['lsr_button'] = False

    if st.button('Run least squares regression'):
        st.session_state['lsr_button'] = True

    #Process data, plot, and start lsr module
    if st.session_state['lsr_button']:
        df = df.loc[:, active_cols]
        df = preprocess(df)

        #Plot heatmap
        st.plotly_chart(corr_plot(df))

        lsr(df, True)