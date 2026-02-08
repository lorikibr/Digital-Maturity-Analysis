import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Digital Maturity Dashboard", layout="wide")

st.title("SME Digital Maturity Assessment")
st.write("Analysis of digital transformation across 1,000 companies using the EDIH framework.")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        before = pd.read_excel("data/rawdma_before.xlsx")
        after = pd.read_excel("data/rawdma_after.xlsx")
        
        # Merging datasets and calculating growth
        merged = pd.merge(before, after, on=["Company_ID", "Company_Name", "Sector", "Country"], suffixes=('_Before', '_After'))
        merged['Maturity_Growth'] = merged['Overall_Maturity_After'] - merged['Overall_Maturity_Before']
        
        return merged, after
    except Exception as e:
        st.error(f"Error: Missing data files. Please run the generator script first. {e}")
        return None, None

df_merged, df_after = load_data()

if df_merged is not None:
    # --- SECTION 1: PERFORMANCE SUMMARY ---
    # Paired T-Test to validate the significance of the digital intervention
    t_stat, p_val = stats.ttest_rel(df_merged['Overall_Maturity_Before'], df_merged['Overall_Maturity_After'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Initial Maturity", f"{df_merged['Overall_Maturity_Before'].mean():.2f}%")
    col2.metric("Average Final Maturity", f"{df_merged['Overall_Maturity_After'].mean():.2f}%")
    col3.metric("Growth Significance (P-Value)", f"{p_val:.4f}")

    st.write("---")

    # --- SECTION 2: LEADERBOARD ---
    st.subheader("Final Maturity Ranking")
    col_best, col_worst = st.columns(2)

    with col_best:
        st.write("Top 10 Performing Companies")
        top_10 = df_merged.nlargest(10, 'Overall_Maturity_After')[['Company_Name', 'Sector', 'Overall_Maturity_After']]
        st.dataframe(top_10, hide_index=True, use_container_width=True)

    with col_worst:
        st.write("Bottom 10 Companies (Support Needed)")
        bot_10 = df_merged.nsmallest(10, 'Overall_Maturity_After')[['Company_Name', 'Sector', 'Overall_Maturity_After']]
        st.dataframe(bot_10, hide_index=True, use_container_width=True)

    st.write("---")

    # --- SECTION 3: DEEP DIVE ANALYSIS ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Maturity Growth by Sector")
        sector_growth = df_merged.groupby('Sector')['Maturity_Growth'].mean().reset_index().sort_values('Maturity_Growth')
        # Using a clean color scale for sector growth
        fig_bar = px.bar(sector_growth, x='Maturity_Growth', y='Sector', orientation='h',
                         title="Average Improvement per Sector",
                         color='Maturity_Growth', color_continuous_scale='Greens')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("Individual Company Footprint")
        company_list = df_merged['Company_Name'].unique()
        selected_company = st.selectbox("Select a company for detailed view:", company_list)
        
        c_data = df_merged[df_merged['Company_Name'] == selected_company].iloc[0]
        
        categories = ["Strategy", "Infrastructure", "HR Skills", "Data Mgmt", "AI & Automation", "Green Tech"]
        before_vals = [c_data['D_Strategy_Before'], c_data['D_Infrastructure_Before'], c_data['D_Human_Centric_Before'], 
                       c_data['D_Data_Mgmt_Before'], c_data['D_Automation_AI_Before'], c_data['D_Green_Digital_Before']]
        after_vals = [c_data['D_Strategy_After'], c_data['D_Infrastructure_After'], c_data['D_Human_Centric_After'], 
                      c_data['D_Data_Mgmt_After'], c_data['D_Automation_AI_After'], c_data['D_Green_Digital_After']]

        fig_radar = go.Figure()

        # Before (Red)
        fig_radar.add_trace(go.Scatterpolar(
            r=before_vals, theta=categories, fill='toself', name='Initial State',
            line_color='rgba(255, 75, 75, 0.8)', fillcolor='rgba(255, 75, 75, 0.3)'
        ))
        # After (Green)
        fig_radar.add_trace(go.Scatterpolar(
            r=after_vals, theta=categories, fill='toself', name='Post-Intervention',
            line_color='rgba(0, 204, 150, 0.9)', fillcolor='rgba(0, 204, 150, 0.4)'
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title=f"Growth Visualization: {selected_company}"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.write("---")

    # --- SECTION 4: AI PREDICTION MODEL ---
    st.subheader("Strategic Maturity Predictor")
    st.write("Simulate how improvements in specific dimensions affect the overall maturity score.")

    # Model training logic
    dims = ['D_Strategy', 'D_Infrastructure', 'D_Human_Centric', 'D_Data_Mgmt', 'D_Automation_AI', 'D_Green_Digital']
    X = df_after[dims]
    y = df_after['Overall_Maturity']
    model = LinearRegression().fit(X, y)

    p_cols = st.columns(3)
    user_inputs = []
    labels = ["Digital Strategy", "Infrastructure", "Staff Skills", "Data Usage", "AI & Robotics", "Green Tech"]
    
    for i, label in enumerate(labels):
        with p_cols[i % 3]:
            val = st.slider(label, 0, 100, 50)
            user_inputs.append(val)

    prediction = model.predict([user_inputs])[0]
    st.info(f"The Predicted Maturity Score for this configuration is: {prediction:.2f}%")

    # Feature Importance
    st.write("Variable Impact (Which factors drive the score most?)")
    importance = pd.DataFrame({'Dimension': labels, 'Impact': model.coef_})
    importance = importance.sort_values(by='Impact', ascending=True)
    fig_imp = px.bar(importance, x='Impact', y='Dimension', orientation='h', color='Impact', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_imp, use_container_width=True)

# Sidebar
st.sidebar.markdown("### Project Info")
st.sidebar.write("Project: Digital Maturity Analysis")
st.sidebar.write("Standard: EDIH Open DMA")
st.sidebar.write("Methodology: CRISP-DM")