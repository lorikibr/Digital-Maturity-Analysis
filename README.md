# Digital Maturity Assessment Tool (EDIH Open DMA)

An end-to-end data science project assessing the digital transformation of 1,000 companies using the CRISP-DM methodology.

## 📊 Methodology (CRISP-DM)
1. **Business Understanding:** Quantifying the ROI of digital transformation interventions.
2. **Data Understanding:** Based on the EDIH (European Digital Innovation Hubs) 6-dimension framework.
3. **Data Preparation:** Synthetic generation of "Before" and "After" datasets with latent variable correlation.
4. **Modeling:** Linear Regression to determine feature importance and predict maturity scores.
5. **Evaluation:** Paired T-Tests to validate statistical significance of improvements.
6. **Deployment:** Interactive Streamlit Dashboard and automated PDF reporting.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Generate data & reports: `python src/data_generator.py`
3. Launch dashboard: `streamlit run src/dashboard.py`

## 📈 Key Visuals
- **Radar Charts:** Individual company performance tracking.
- **Correlation Heatmaps:** Identifying drivers of digital growth.
- **AI Predictor:** Real-time maturity estimation.
