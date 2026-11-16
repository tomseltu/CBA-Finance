import pandas as pd
import numpy as np
import os

from preprocessing import load_all_datasets, get_continent, get_funder_category

# ------------------- Pfade -------------------
vulnerability_path = "data/vulnerability.csv"
population_path = "data/population.csv"
crdf_path = "data/CRDF-RP.csv"

FINANCE_COL = 'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand'

# SIDS-Liste ist in CONTINENT_MAPPING NICHT separat vorhanden (läuft unter "Oceania"/"North America").
# Für Tabelle D.3 daher manuell kuratiert, analog zur bisherigen Annex-Diskussion.
SIDS_COUNTRIES = [
    'Kiribati', 'Marshall Islands', 'Micronesia, Federated States of', 'Tonga', 'Tuvalu', 'Vanuatu'
]

if not os.path.exists('output'):
    os.makedirs('output')

print("Lade Datensätze...")
vulnerability_df, population_df, crdf_df = load_all_datasets(
    vulnerability_path, population_path, crdf_path
)
print(f"CRDF (CBA-gefiltert) shape: {crdf_df.shape}")

# =====================================================================
# D.1 — CBA Finance nach Funder Category und Region (2011– aktuell)
# =====================================================================
print("\n=== D.1: Funder Category x Region ===")

d1 = crdf_df.copy()
d1['Amount'] = pd.to_numeric(d1[FINANCE_COL], errors='coerce')
d1 = d1[d1['Amount'] > 0]

table_d1 = d1.pivot_table(
    index='Funder Category',
    columns='Region',
    values='Amount',
    aggfunc='sum',
    fill_value=0
) / 1000  # Thousand -> Million USD

table_d1['Total'] = table_d1.sum(axis=1)
table_d1.loc['Total'] = table_d1.sum(axis=0)
table_d1 = table_d1.round(1)

print(table_d1)
table_d1.to_csv('output/annex_D1_funder_x_region.csv')
print("Gespeichert: output/annex_D1_funder_x_region.csv")
print("Einheit: Million USD (constant 2023 US$, wie CRDF-RP-Exportspalte)")

# =====================================================================
# D.2 — Financial Instrument Verteilung, ausgewählte Jahre
# =====================================================================
print("\n=== D.2: Financial Instrument nach Jahr (USD Mio. und Anteil %) ===")

selected_years = [2011, 2015, 2018, 2021, 2023]

d2 = crdf_df.copy()
d2 = d2[d2['Financial Instrument'].notna()]
d2['Amount'] = pd.to_numeric(d2[FINANCE_COL], errors='coerce')
d2 = d2[d2['Amount'] > 0]
d2 = d2[d2['Year'].isin(selected_years)]

grouped = d2.groupby(['Year', 'Financial Instrument'])['Amount'].sum().reset_index()
grouped['Amount_Mio'] = grouped['Amount'] / 1000

year_totals = grouped.groupby('Year')['Amount_Mio'].sum()

rows = []
for instrument in sorted(grouped['Financial Instrument'].unique()):
    row = {'Financial Instrument': instrument}
    for year in selected_years:
        sub = grouped[(grouped['Year'] == year) & (grouped['Financial Instrument'] == instrument)]
        amount = sub['Amount_Mio'].sum() if not sub.empty else 0.0
        total = year_totals.get(year, np.nan)
        share = (amount / total * 100) if total and total > 0 else 0.0
        row[f'{year}_USDm'] = round(amount, 1)
        row[f'{year}_pct'] = round(share, 1)
    rows.append(row)

table_d2 = pd.DataFrame(rows)
print(table_d2.to_string(index=False))
table_d2.to_csv('output/annex_D2_financial_instrument.csv', index=False)
print("Gespeichert: output/annex_D2_financial_instrument.csv")

# =====================================================================
# D.3 — ND-GAIN Vulnerability (all-period mean) + CBA Finance per Capita
# =====================================================================
print("\n=== D.3: Vulnerability (Mittelwert aller Jahre) + CBA Finance per Capita ===")

# Finance pro Land, aktueller Stand (Summe über gesamten Zeitraum)
finance_per_country = crdf_df[crdf_df['Recipient'] != 'REGIONAL'].groupby('Recipient').agg(
    Total_Finance=(FINANCE_COL, 'sum')
).reset_index()
finance_per_country.columns = ['Name', 'Total_Finance']
finance_per_country['Total_Finance'] = pd.to_numeric(
    finance_per_country['Total_Finance'], errors='coerce'
) * 1000  # Thousand -> USD
finance_per_country = finance_per_country[finance_per_country['Total_Finance'] > 0]

# Population: einzelner Jahreswert 2023 (wie im Scatter-Plot-Code, NICHT Mehrjahresdurchschnitt)
latest_year_col = '2023 [YR2023]'
population_latest = population_df[['Country Name', latest_year_col]].copy()
population_latest.columns = ['Name', 'Population']
population_latest['Population'] = pd.to_numeric(population_latest['Population'], errors='coerce')

d3 = finance_per_country.merge(population_latest, on='Name', how='left')
d3['Finance_per_capita_avg_annual'] = (d3['Total_Finance'] / d3['Population']) / 13  # /Jahre im Analysezeitraum

# Vulnerability: Mittelwert über alle verfügbaren Jahresspalten (all-period average, NICHT "most recent")
year_cols = [c for c in vulnerability_df.columns if c.isdigit()]
vuln_avg = vulnerability_df[['Name'] + year_cols].copy()
for c in year_cols:
    vuln_avg[c] = pd.to_numeric(vuln_avg[c], errors='coerce')
vuln_avg['Vulnerability_Index'] = vuln_avg[year_cols].mean(axis=1)
vuln_avg = vuln_avg[['Name', 'Vulnerability_Index']]

d3 = d3.merge(vuln_avg, on='Name', how='left')
d3 = d3.dropna(subset=['Vulnerability_Index', 'Finance_per_capita_avg_annual'])

# SIDS-Flag (manuell kuratiert, s.o. — kein eigenes CONTINENT_MAPPING-Feld)
d3['Is_SIDS'] = d3['Name'].isin(SIDS_COUNTRIES)

d3_sorted = d3.sort_values('Vulnerability_Index', ascending=False)
table_d3 = d3_sorted[['Name', 'Vulnerability_Index', 'Finance_per_capita_avg_annual', 'Is_SIDS']].round(3)

print(table_d3.head(20).to_string(index=False))
table_d3.to_csv('output/annex_D3_vulnerability_percapita.csv', index=False)
print("Gespeichert: output/annex_D3_vulnerability_percapita.csv")
print("Hinweis: Finance per capita = Summe CBA-Finance (gesamter Analysezeitraum) / Population(2023) / 13 Jahre")
print("Hinweis: Is_SIDS ist eine manuell kuratierte Teilmenge, nicht Teil von CONTINENT_MAPPING")

print("\n=== Fertig. Alle Annex-D-Tabellen liegen unter output/ ===")