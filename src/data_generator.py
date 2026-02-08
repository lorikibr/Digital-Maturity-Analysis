import pandas as pd
import numpy as np
import random
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# --- CONFIGURATION ---
NUM_RECORDS = 1000
SECTORS = ['Manufacturing', 'Retail', 'Healthcare', 'Logistics', 'Finance']
DIMENSIONS = ['Strategy', 'Infrastructure', 'Human_Centric', 'Data_Mgmt', 'Automation_AI', 'Green_Digital']
WEIGHTS = [0.25, 0.15, 0.10, 0.20, 0.20, 0.10] # Strategic weighting

def generate_dma_data(n=1000, state="before"):
    np.random.seed(42 if state == "before" else 24)
    companies = [f"Company_{i:04d}" for i in range(1, n + 1)]
    
    # Latent Variable: Companies generally have a baseline capability
    base_capability = np.random.normal(loc=0.35 if state=="before" else 0.72, scale=0.12, size=n)
    base_capability = np.clip(base_capability, 0.1, 0.95)
    
    data = {
        'Company_ID': companies,
        'Company_Name': [f"{c} Inc." for c in companies],
        'Sector': [random.choice(SECTORS) for _ in range(n)],
        'Country': ['North Macedonia' for _ in range(n)]
    }
    
    # Generate dimensions based on base + noise
    dim_data = []
    for dim in DIMENSIONS:
        noise = np.random.normal(0, 0.08, n)
        val = (base_capability + noise) * 100
        data[f"D_{dim}"] = np.clip(val, 0, 100).astype(int)
        dim_data.append(data[f"D_{dim}"])
    
    df = pd.DataFrame(data)
    # Calculate weighted maturity
    df['Overall_Maturity'] = np.average(np.array(dim_data).T, axis=1, weights=WEIGHTS).round(2)
    return df

# Create Folders if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("Generating Master-level datasets...")
df_before = generate_dma_data(NUM_RECORDS, state="before")
df_after = generate_dma_data(NUM_RECORDS, state="after")

# Ensure 'After' is logically an improvement over 'Before'
df_after['Overall_Maturity'] = np.maximum(df_after['Overall_Maturity'], df_before['Overall_Maturity'] + 5)

df_before.to_excel("data/rawdma_before.xlsx", index=False)
df_after.to_excel("data/rawdma_after.xlsx", index=False)

# --- PDF REPORTING ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Digital Maturity Assessment - EDIH Framework', 0, 1, 'C')
        self.ln(5)

def create_radar_chart(comp_id, row_b, row_a):
    categories = DIMENSIONS
    v_b = [row_b[f'D_{d}'] for d in DIMENSIONS] + [row_b[f'D_{DIMENSIONS[0]}']]
    v_a = [row_a[f'D_{d}'] for d in DIMENSIONS] + [row_a[f'D_{DIMENSIONS[0]}']]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, v_b, color='red', label='Initial')
    ax.fill(angles, v_b, color='red', alpha=0.1)
    ax.plot(angles, v_a, color='green', label='Post-Intervention')
    ax.fill(angles, v_a, color='green', alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))
    path = f"reports/chart_{comp_id}.png"
    plt.savefig(path)
    plt.close()
    return path

print("Generating 5 sample reports in /reports folder...")
merged = pd.merge(df_before, df_after, on="Company_ID", suffixes=('_B', '_A'))
for i, row in merged.head(5).iterrows():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Company: {row['Company_Name_A']} | Sector: {row['Sector_A']}", ln=1)
    
    # Table
    pdf.set_font("Arial", 'B', 10)
    cols = ["Dimension", "Before", "After", "Delta"]
    for c in cols: pdf.cell(40, 10, c, 1)
    pdf.ln()
    pdf.set_font("Arial", size=10)
    for d in DIMENSIONS:
        b, a = row[f"D_{d}_B"], row[f"D_{d}_A"]
        pdf.cell(40, 10, d, 1)
        pdf.cell(40, 10, str(b), 1)
        pdf.cell(40, 10, str(a), 1)
        pdf.cell(40, 10, f"+{a-b}", 1)
        pdf.ln()
    
    img_path = create_radar_chart(row['Company_ID'], df_before.iloc[i], df_after.iloc[i])
    pdf.image(img_path, x=50, y=110, w=110)
    os.remove(img_path)
    pdf.output(f"reports/Report_{row['Company_ID']}.pdf")

print("Done.")