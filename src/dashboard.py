import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Digital Maturity Dashboard", layout="wide")

st.title("Digital Maturity Assessment Dashboard")
st.write("Comparison of company performance before and after digital intervention.")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Loading files from the project folder structure
        before = pd.read_excel("data/rawdma_before.xlsx")
        after = pd.read_excel("data/rawdma_after.xlsx")
        
        # Merging datasets to calculate the growth delta
        merged = pd.merge(before, after, on=["Company_ID", "Company_Name", "Sector", "Country"], suffixes=('_Before', '_After'))
        merged['Maturity_Growth'] = merged['Overall_Maturity_After'] - merged['Overall_Maturity_Before']
        
        return merged, after
    except Exception as e:
        st.error(f"Error: Ensure data files exist in the /data folder. {e}")
        return None, None

df_merged, df_after = load_data()

if df_merged is not None:
    # --- SECTION 1: KEY PERFORMANCE INDICATORS ---
    # Performing a paired t-test to validate the significance of results
    t_stat, p_val = stats.ttest_rel(df_merged['Overall_Maturity_Before'], df_merged['Overall_Maturity_After'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Maturity (Initial)", f"{df_merged['Overall_Maturity_Before'].mean():.2f}")
    col2.metric("Average Maturity (Final)", f"{df_merged['Overall_Maturity_After'].mean():.2f}")
    col3.metric("Average Growth", f"{df_merged['Maturity_Growth'].mean():.2f}")

    st.write("---")

    # --- SECTION 2: LEADERBOARD ---
    st.subheader("Top and Bottom Performers (Final Results)")
    col_best, col_worst = st.columns(2)

    with col_best:
        st.write("Top 10 Cases")
        top_10 = df_merged.nlargest(10, 'Overall_Maturity_After')[['Company_Name', 'Sector', 'Overall_Maturity_After']]
        st.dataframe(top_10, hide_index=True, use_container_width=True)

    with col_worst:
        st.write("Bottom 10 Cases")
        bot_10 = df_merged.nsmallest(10, 'Overall_Maturity_After')[['Company_Name', 'Sector', 'Overall_Maturity_After']]
        st.dataframe(bot_10, hide_index=True, use_container_width=True)

    st.write("---")

    # --- SECTION 3: DATA ANALYSIS ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Growth by Business Sector")
        sector_growth = df_merged.groupby('Sector')['Maturity_Growth'].mean().reset_index()
        fig_bar = px.bar(sector_growth, x='Sector', y='Maturity_Growth', color='Sector', 
                         title="Mean Maturity Improvement per Sector")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("Variable Correlations")
        # Selecting numeric columns for the final state
        dims_cols = [c for c in df_merged.columns if "_After" in c]
        corr = df_merged[dims_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                             title="Correlation Matrix of Digital Dimensions")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.write("---")

    # --- SECTION 4: PREDICTIVE MODELING (LINEAR REGRESSION) ---
    st.subheader("Maturity Prediction Model")
    st.write("Adjust the input variables to predict the overall digital maturity level.")

    # Model training using the 6 core dimensions
    dims = ['D_Strategy', 'D_Infrastructure', 'D_Human_Centric', 'D_Data_Mgmt', 'D_Automation_AI', 'D_Green_Digital']
    X = df_after[dims]
    y = df_after['Overall_Maturity']
    model = LinearRegression()
    model.fit(X, y)

    # Creating sliders for the prediction interface
    p_cols = st.columns(3)
    user_inputs = []
    
    # Mapping to human-readable labels
    labels = ["Strategy", "Infrastructure", "Human Resources", "Data Management", "Automation & AI", "Green Tech"]
    
    for i, label in enumerate(labels):
        with p_cols[i % 3]:
            val = st.slider(label, 0, 100, 50)
            user_inputs.append(val)

    # Prediction result
    prediction = model.predict([user_inputs])[0]
    st.info(f"The Predicted Overall Maturity Score is: {prediction:.2f}")

    # Showing Feature Importance (Coefficients)
    st.write("Variable Importance (Linear Regression Coefficients)")
    importance = pd.DataFrame({'Dimension': labels, 'Impact': model.coef_})
    importance = importance.sort_values(by='Impact', ascending=False)
    fig_imp = px.bar(importance, x='Impact', y='Dimension', orientation='h')
    st.plotly_chart(fig_imp, use_container_width=True)

# Footer info for project context
st.sidebar.markdown("### Project Metadata")
st.sidebar.write("Methodology: CRISP-DM")
st.sidebar.write("Framework: EDIH DMA")