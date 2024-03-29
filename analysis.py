import numpy as np
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.figure_factory as ff
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sympy import Symbol, Eq, latex


# Read & drop unused columns
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
    
    # Display information about 3D visualization
    st.markdown('</br>', unsafe_allow_html=True)
    st.info("INFO: On selecting any 2 parameters, a 3D visualization of the least squares algorithm is provided")

    return st.session_state['active_cols']

def preprocess(data):
    # Normalize continuous variables (if they exist)
    # to_normalize = ['Age', 'SibSp', 'Parch', 'Fare', 'PClass']
    # to_normalize = [c for c in to_normalize if c in data.columns]
    # if to_normalize:
    #     data[to_normalize] = MinMaxScaler().fit_transform(data[to_normalize])  

    # woman = 0, man = 1
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
    
    # split training/ testing sets 75/25
    f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.25, random_state=1)
    
    # Fit & evaluate model
    m = LinearRegression()
    m.fit(f_train, t_train)
    pred = m.predict(f_test)    
    ms = mean_squared_error(t_test, pred)
    r2 = r2_score(t_test, pred)

    st.markdown(f"""
        <p style='font-size: 25px;'>
            Mean Squared Error: <strong>{round(ms, 2)}</strong><br>
            r<sup>2</sup> score: <strong>{round(r2, 2)}</strong>
        </p>
        """, unsafe_allow_html=True)

    # Survival Score Prediction
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
        st.markdown(f"""
        <p style='font-size: 25px;'>
            Predicted survival value: <strong>{round(output[0], 2)}</strong><br>
        </p>
        """, unsafe_allow_html=True)
    
    # Plot 3D visualization if exactly 2 features are selected
    if len(features.columns) == 2:
        st.markdown('</br>', unsafe_allow_html=True)
        
        import plotly.express as px

        fig = px.scatter_3d(
            x=f_test.iloc[:, 0], 
            y=f_test.iloc[:, 1], 
            z=pred,
            labels={'x': features.columns[0], 'y': features.columns[1], 'z': 'Survival Probability'},
            title='3D Scatter Plot',
            opacity=0.7
        )
        
        A, B, C = m.coef_[0], m.coef_[1], m.intercept_

        # Create meshgrid
        x_surf = np.linspace(f_test.iloc[:, 0].min(), f_test.iloc[:, 0].max(), 10)
        y_surf = np.linspace(f_test.iloc[:, 1].min(), f_test.iloc[:, 1].max(), 10)
        x_surf, y_surf = np.meshgrid(x_surf, y_surf)
        z_surf = A * x_surf + B * y_surf + C

        # Add plane to the plot
        fig.add_trace(go.Surface(
            x=x_surf, 
            y=y_surf, 
            z=z_surf,
            colorscale='blues',
            opacity=0.5,
            name='Plane'
        ))

        # Set plot layout
        fig.update_layout(
            scene=dict(
                xaxis_title=features.columns[0],
                yaxis_title=features.columns[1],
                zaxis_title='Survival Probability'
            ),
            title='Titanic Survival Probability Visualization'
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=True, width=800, height=600)

def description():
    # Display mathematical formulation using LaTeX
    st.markdown('</br></br>', unsafe_allow_html=True)
    st.header('Linear Regression (Fix)')

    st.markdown(r'''
    **Least Squares Algorithm Overview:**
    $$\min_{\beta} \sum_{i=1}^{n} (y_i - X_i \cdot \beta)^2$$

    The least squares algorithm minimizes the sum of squared errors between predicted (\(y_i\)) and actual (\(X_i \cdot \beta\)) values.

    **Application to Titanic Data:**

    1. **Data Preprocessing**: Handle missing values, encode categorical variables, and normalize features if necessary.

    2. **Feature Selection**: Choose features like class, age, siblings/spouses, parents/children, and fare.

    3. **Model Fitting**: Split data, fit a linear regression model, and evaluate using metrics like Mean Squared Error (MSE) and \( R^2 \) score.
    $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 $$
    $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$

    4. **Prediction**: Predict survival probability based on user inputs.

    5. **Visualization**: Plot data points and the fitted plane in a 3D scatter plot.
    ''')


if __name__ == '__main__':
    # Setup app
    st.title('The Unsinkable Ship: Who survived the Titanic?')
    st.image("titanic_drawing.jpeg", caption="Titanic Sinking, Willy Stöwer. Wikimedia Commons.", use_column_width=True)
    st.write('The [Titanic](https://www.history.com/topics/early-20th-century-us/titanic#unsinkable-titanic-s-fatal-flaws), deemed "practically unsinkable" by experts, sunk in 1912. More than 1,500 of the 2,240 passengers onboard were lost. Here, we use passenger data to explore the factors that contributed to survival. Select features to see how well they predict survival via least squares regression.')
    
    # Select columns
    st.header('Select features')            
    active_cols = select_features()

    if 'lsr_button' not in st.session_state:
        st.session_state['lsr_button'] = False

    if st.button('Run least squares regression'):
        st.session_state['lsr_button'] = True

    # Process data, plot, and start lsr module
    if st.session_state['lsr_button']:
        df = df.loc[:, active_cols]
        df = preprocess(df)

        # Plot heatmap
        st.plotly_chart(corr_plot(df))

        lsr(df, True)
    
    description()
