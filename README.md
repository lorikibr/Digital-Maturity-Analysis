# Digital Maturity Assessment Tool

A data science project that simulates and analyzes technology adoption across 1,000 companies. This tool tracks progress between two stages (Before and After) and identifies key growth drivers.

## Project Steps (CRISP-DM)
1. **Business Goal:** Measure the impact of digital improvements on total performance.
2. **Categories:** Strategy, Infrastructure, Staff Skills, Data Management, AI, and Green Tech.
3. **Data:** Generated 1,000 company records with "Before" and "After" scores for analysis.
4. **Modeling:** Used Linear Regression to calculate which categories have the most impact on the final score.
5. **Validation:** Used a T-Test to confirm that the improvement in scores is statistically significant.
6. **Deployment:** Created an interactive dashboard and an automated PDF reporting system.

## How to Run
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Generate data & reports:** `python src/data_generator.py`
3. **Launch dashboard:** `python -m streamlit run src/dashboard.py`

## Features
- **Comparison Charts:** Radar plots showing the "footprint" of a company's growth.
- **Performance Ranking:** Tables showing the top and bottom performing companies.
- **Score Predictor:** A tool to estimate scores based on custom inputs.
- **Statistical Results:** Displays the P-Value to verify data reliability.
