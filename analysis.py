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
df = pd.read_csv('happiness.csv')
df.drop(columns=['upperwhisker', 'lowerwhisker', 'Dystopia + residual'])
vars = {
        "GDP per capita": "Explained by: Log GDP per capita",
        "Social Support": "Explained by: Social support",
        "Life Expectancy": "Explained by: Healthy life expectancy",
        "Freedom of Choice": "Explained by: Freedom to make life choices",
        "Generosity": "Explained by: Generosity",
        "Perceived Corruption": "Explained by: Perceptions of corruption"
    }

def select_features():
    if 'active_cols' not in st.session_state:
        st.session_state['active_cols'] = ['Ladder score']

    for human_var, df_var in vars.items():
        if st.checkbox(human_var, key=df_var):
            if df_var not in st.session_state['active_cols']:
                st.session_state['active_cols'].append(df_var)
        else:
            if df_var in st.session_state['active_cols']:
                st.session_state['active_cols'].remove(df_var)
    
    # Display information about 3D visualization
    st.markdown('</br>', unsafe_allow_html=True)
    st.info("On selecting any 2 parameters, a 3D visualization of the least squares algorithm is provided")

    return st.session_state['active_cols']

def preprocess(data):
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
    features = data.drop('Ladder score', axis=1)
    target = data['Ladder score']
    
    # split training/ testing sets 75/25
    f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.25, random_state=1)
    
    # Fit & evaluate model
    m = LinearRegression()
    m.fit(f_train, t_train)
    pred = m.predict(f_test)    
    ms = mean_squared_error(t_test, pred)
    r2 = r2_score(t_test, pred)

    # Mean Squared Error: <strong>{round(ms, 2)}</strong><br>
    st.markdown(f"""
        <p style='font-size: 25px;'>
            r<sup>2</sup> score: <strong>{round(r2, 2)}</strong>
        </p>
        """, unsafe_allow_html=True)

    # Survival Score Prediction
    st.header('Predicted Happiness by Features')
    st.write('Provide values between 0-2 to test how the quality of various features impact happiness.')    
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
            Predicted happiness index: <strong>{round(output[0], 2)}</strong><br>
        </p>
        """, unsafe_allow_html=True)
    
    # Plot 3D visualization if exactly 2 features are selected
    if len(features.columns) == 2:
        st.markdown('</br>', unsafe_allow_html=True)
        
        import plotly.express as px

        fig = px.scatter_3d(
            x=features.iloc[:, 0], 
            y=features.iloc[:, 1], 
            z=target,
            labels={'x': features.columns[0], 'y': features.columns[1], 'z': 'Predicted Happiness (1-10)'},
            title='3D Scatter Plot',
            opacity=0.7
        )
        
        A, B, C = m.coef_[0], m.coef_[1], m.intercept_

        # Create meshgrid
        x_surf = np.linspace(features.iloc[:, 0].min(), features.iloc[:, 0].max(), 10)
        y_surf = np.linspace(features.iloc[:, 1].min(), features.iloc[:, 1].max(), 10)
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
                zaxis_title='Predicted Happiness',

                xaxis_range=[features.iloc[:, 0].min(), features.iloc[:, 0].max()],
                yaxis_range=[features.iloc[:, 1].min(), features.iloc[:, 1].max()],
                zaxis_range=[target.min(), target.max()],
            ),
            title='Predicted Happiness Visualization'
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

    **Application to Happiness Data:**

    1. **Data Preprocessing**: Handle missing values, encode categorical variables, and normalize features if necessary.

    2. **Feature Selection**: Choose features like class, age, siblings/spouses, parents/children, and fare.

    3. **Model Fitting**: Split data, fit a linear regression model, and evaluate using metrics like Mean Squared Error (MSE) and \( R^2 \) score.
    $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 $$
    $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$

    4. **Prediction**: Predict survival probability based on user inputs.

    5. **Visualization**: Plot data points and the fitted plane in a 3D scatter plot.
    ''')


if __name__ == '__main__':
    st.title('The Happiness Formula: What features predict a country\'s happiness?')
    st.image("img.png", caption="Attribution: Robert Collins, \"Football outside Jakarta\"", use_column_width=True)
    st.write("Use the interactive tools below to model how various features predict happiness worldwide.")

    st.header('Select features')            
    active_cols = select_features()

    if 'lsr_button' not in st.session_state:
        st.session_state['lsr_button'] = False

    if st.button('Run least squares regression'):
        st.session_state['lsr_button'] = True

    if st.session_state['lsr_button']:
        df = df.loc[:, active_cols]
        df = preprocess(df)

        st.plotly_chart(corr_plot(df))

        lsr(df, True)
    
    description()
