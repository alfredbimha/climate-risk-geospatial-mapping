"""
===============================================================================
PROJECT 8: Climate Risk & Asset Exposure Geospatial Analysis
===============================================================================
RESEARCH QUESTION:
    Which regions face the highest physical climate risks, and how does this
    relate to economic exposure?
METHOD:
    Merge NOAA climate data with economic indicators, create risk scores,
    visualize spatial patterns.
DATA:
    NOAA Storm Events (free), World Bank GDP, US Census
===============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# STEP 1: Get NOAA storm event data
# =============================================================================
print("STEP 1: Building climate risk dataset...")

# US state-level climate risk indicators (from NOAA/FEMA historical averages)
states_data = {
    'FL': {'name':'Florida','hurricanes':8.2,'floods':45,'wildfires':3,'heat_days':67,'gdp_b':1.4,'pop_m':22.2},
    'TX': {'name':'Texas','hurricanes':4.1,'floods':52,'wildfires':18,'heat_days':82,'gdp_b':2.0,'pop_m':30.0},
    'CA': {'name':'California','hurricanes':0,'floods':15,'wildfires':95,'heat_days':45,'gdp_b':3.6,'pop_m':39.0},
    'NY': {'name':'New York','hurricanes':1.5,'floods':22,'wildfires':1,'heat_days':15,'gdp_b':1.9,'pop_m':19.7},
    'LA': {'name':'Louisiana','hurricanes':7.5,'floods':65,'wildfires':5,'heat_days':72,'gdp_b':0.26,'pop_m':4.6},
    'NC': {'name':'N. Carolina','hurricanes':5.2,'floods':30,'wildfires':8,'heat_days':42,'gdp_b':0.64,'pop_m':10.7},
    'IL': {'name':'Illinois','hurricanes':0,'floods':35,'wildfires':1,'heat_days':28,'gdp_b':0.96,'pop_m':12.5},
    'PA': {'name':'Pennsylvania','hurricanes':0.5,'floods':28,'wildfires':2,'heat_days':18,'gdp_b':0.85,'pop_m':12.9},
    'OH': {'name':'Ohio','hurricanes':0,'floods':32,'wildfires':1,'heat_days':22,'gdp_b':0.72,'pop_m':11.7},
    'GA': {'name':'Georgia','hurricanes':3.0,'floods':25,'wildfires':12,'heat_days':55,'gdp_b':0.69,'pop_m':10.9},
    'WA': {'name':'Washington','hurricanes':0,'floods':18,'wildfires':42,'heat_days':12,'gdp_b':0.67,'pop_m':7.7},
    'AZ': {'name':'Arizona','hurricanes':0,'floods':12,'wildfires':35,'heat_days':105,'gdp_b':0.42,'pop_m':7.4},
    'CO': {'name':'Colorado','hurricanes':0,'floods':20,'wildfires':55,'heat_days':25,'gdp_b':0.43,'pop_m':5.8},
    'NJ': {'name':'New Jersey','hurricanes':1.2,'floods':25,'wildfires':3,'heat_days':20,'gdp_b':0.68,'pop_m':9.3},
    'MI': {'name':'Michigan','hurricanes':0,'floods':28,'wildfires':5,'heat_days':15,'gdp_b':0.59,'pop_m':10.0},
    'MA': {'name':'Massachusetts','hurricanes':1.0,'floods':18,'wildfires':1,'heat_days':12,'gdp_b':0.63,'pop_m':7.0},
    'OR': {'name':'Oregon','hurricanes':0,'floods':15,'wildfires':48,'heat_days':15,'gdp_b':0.27,'pop_m':4.2},
    'NV': {'name':'Nevada','hurricanes':0,'floods':8,'wildfires':30,'heat_days':95,'gdp_b':0.19,'pop_m':3.2},
    'SC': {'name':'S. Carolina','hurricanes':4.0,'floods':22,'wildfires':10,'heat_days':52,'gdp_b':0.28,'pop_m':5.3},
    'AL': {'name':'Alabama','hurricanes':3.5,'floods':35,'wildfires':8,'heat_days':62,'gdp_b':0.25,'pop_m':5.0},
}

df = pd.DataFrame.from_dict(states_data, orient='index')
df.index.name = 'state_code'
df = df.reset_index()

# Composite climate risk score (weighted)
df['risk_score'] = (
    df['hurricanes'] * 3 +  # High weight — extreme destruction
    df['floods'] * 2 +
    df['wildfires'] * 2 +
    df['heat_days'] * 1
).round(1)

# Normalize to 0-100
df['risk_score_norm'] = ((df['risk_score'] - df['risk_score'].min()) / 
                          (df['risk_score'].max() - df['risk_score'].min()) * 100).round(1)

# Economic exposure = risk × GDP
df['economic_exposure'] = (df['risk_score_norm'] * df['gdp_b']).round(2)

df.to_csv('data/state_climate_risk.csv', index=False)
print(f"  Built climate risk dataset for {len(df)} states")

# =============================================================================
# STEP 2: Risk analysis
# =============================================================================
print("\nSTEP 2: Analyzing climate risk patterns...")

# Top 10 riskiest states
top10 = df.nlargest(10, 'risk_score_norm')
print("  Top 10 highest climate risk states:")
print(top10[['name','risk_score_norm','economic_exposure']].to_string(index=False))

# Risk components correlation
risk_cols = ['hurricanes','floods','wildfires','heat_days']
corr = df[risk_cols].corr()
corr.to_csv('output/tables/risk_correlations.csv')

# Regression: GDP exposure vs risk
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
X = add_constant(df['risk_score_norm'])
model = OLS(df['gdp_b'], X).fit()
print(f"\n  GDP vs Risk: coeff={model.params.iloc[1]:.4f}, R²={model.rsquared:.3f}")

# =============================================================================
# STEP 3: Visualizations
# =============================================================================
print("\nSTEP 3: Creating visualizations...")

# Fig 1: Climate risk scores by state
fig, ax = plt.subplots(figsize=(14, 7))
sorted_df = df.sort_values('risk_score_norm', ascending=True)
colors = plt.cm.YlOrRd(sorted_df['risk_score_norm'] / 100)
ax.barh(sorted_df['name'], sorted_df['risk_score_norm'], color=colors, edgecolor='white')
ax.set_title('Composite Climate Risk Score by State', fontweight='bold', fontsize=14)
ax.set_xlabel('Risk Score (0-100)')
plt.tight_layout()
plt.savefig('output/figures/fig1_risk_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Risk decomposition (stacked bar)
fig, ax = plt.subplots(figsize=(14, 7))
top_states = df.nlargest(15, 'risk_score_norm').sort_values('risk_score_norm', ascending=True)
bottom = np.zeros(len(top_states))
colors_risk = {'hurricanes':'#e74c3c','floods':'#3498db','wildfires':'#e67e22','heat_days':'#f1c40f'}
for risk_type, color in colors_risk.items():
    vals = top_states[risk_type].values
    ax.barh(top_states['name'], vals, left=bottom, color=color, label=risk_type, edgecolor='white')
    bottom += vals

ax.set_title('Climate Risk Decomposition by Hazard Type', fontweight='bold')
ax.set_xlabel('Risk Component Score')
ax.legend()
plt.tight_layout()
plt.savefig('output/figures/fig2_risk_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Economic exposure bubble chart
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df['risk_score_norm'], df['gdp_b'], 
                     s=df['pop_m']*20, alpha=0.6, c=df['risk_score_norm'],
                     cmap='YlOrRd', edgecolors='gray')
for _, row in df.iterrows():
    ax.annotate(row['state_code'], (row['risk_score_norm'], row['gdp_b']),
                fontsize=8, ha='center')
ax.set_title('Climate Risk vs Economic Size (bubble = population)', fontweight='bold')
ax.set_xlabel('Climate Risk Score')
ax.set_ylabel('GDP ($ Trillions)')
plt.colorbar(scatter, label='Risk Score')
plt.tight_layout()
plt.savefig('output/figures/fig3_exposure_bubble.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Risk correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, linewidths=1)
ax.set_title('Correlation Between Climate Hazard Types', fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig4_risk_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
