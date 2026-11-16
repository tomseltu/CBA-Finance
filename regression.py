import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import pycountry
from scipy.stats import pearsonr

from preprocessing import load_all_datasets

# UTF-8 Ausgabe, sonst Abbruch bei Umleitung in Datei (Windows cp1252)
sys.stdout.reconfigure(encoding='utf-8')

# ------------------- Pfade -------------------
vulnerability_path = "data/vulnerability.csv"
population_path = "data/population.csv"
crdf_path = "data/CRDF-RP.csv"
bimts_path = "data/trade.csv"

# ------------------- 1. Hauptdatensätze -------------------
print("Lade CRDF, Vulnerability und Population...")
vulnerability_df, population_df, crdf_df = load_all_datasets(
    vulnerability_path, population_path, crdf_path
)
print(f"CRDF shape: {crdf_df.shape}")
print(f"Vulnerability shape: {vulnerability_df.shape}")

# ------------------- 1b. Vulnerability – jährliche Werte -------------------
year_cols = [col for col in vulnerability_df.columns if col.isdigit()]
vuln_long = vulnerability_df[['Name'] + year_cols].copy()
for col in year_cols:
    vuln_long[col] = pd.to_numeric(vuln_long[col], errors='coerce')
vuln_long = vuln_long.melt(id_vars=['Name'], value_vars=year_cols,
                           var_name='refYear', value_name='Vulnerability_Year')
vuln_long['refYear'] = pd.to_numeric(vuln_long['refYear'], errors='coerce')
vuln_long.rename(columns={'Name': 'Country'}, inplace=True)
vuln_long.dropna(subset=['Vulnerability_Year'], inplace=True)

vuln_countries = set(vuln_long['Country'].dropna())
print(f"Vulnerability-Länder mit Jahreswerten: {len(vuln_countries)}")
# ------------------- 2. BIMTS bilateral mit Trade Volume (Chunk-Verarbeitung) -------------------
print("\nLade BIMTS-Daten (Chunk-Verarbeitung)...")

cols_needed = [
    'PRODUCT_TYPE', 'PRODUCT_HS', 'TRADE_FLOW',
    'OBS_VALUE', 'UNIT_MULT', 'REF_AREA',
    'COUNTERPART_AREA', 'TIME_PERIOD'
]
dtype_dict = {
    'PRODUCT_TYPE': 'str',
    'PRODUCT_HS': 'str',
    'TRADE_FLOW': 'str',
    'REF_AREA': 'str',
    'COUNTERPART_AREA': 'str',
    'TIME_PERIOD': 'str',
    'OBS_VALUE': 'float64',
    'UNIT_MULT': 'float64'
}

chunksize = 500000  # 500k Zeilen pro Chunk
chunks = []

for chunk in pd.read_csv(
    bimts_path,
    sep=",",
    encoding="utf-8",
    low_memory=False,
    usecols=cols_needed,
    dtype=dtype_dict,
    chunksize=chunksize
):
    # Nur Gesamthandel (C + _T) und beide Handelsrichtungen (X, M) behalten
    filtered = chunk[
        (chunk['PRODUCT_TYPE'] == 'C') &
        (chunk['PRODUCT_HS'] == '_T') &
        (chunk['TRADE_FLOW'].isin(['X', 'M']))
    ]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"  Chunk geladen ({len(filtered)} relevante Zeilen)")

if not chunks:
    raise ValueError("Keine relevanten Daten gefunden.")

trade_filtered = pd.concat(chunks, ignore_index=True)
del chunks
print(f"Gesamt nach Filter (C + _T + X/M): {len(trade_filtered)} Zeilen")

# OBS_VALUE_USD berechnen
trade_filtered['OBS_VALUE_USD'] = trade_filtered['OBS_VALUE'] * (10 ** trade_filtered['UNIT_MULT'])

# ISO-Codes → Ländernamen
def iso_to_country(iso):
    try:
        return pycountry.countries.get(alpha_3=iso).name
    except:
        return None

trade_filtered['Donor'] = trade_filtered['REF_AREA'].apply(iso_to_country)
trade_filtered['Recipient'] = trade_filtered['COUNTERPART_AREA'].apply(iso_to_country)

# pycountry-Namen an die OECD/CRDF-Namen angleichen, sonst fallen diese Länder beim Merge raus
PYCOUNTRY_TO_CRDF = {
    'Viet Nam': 'Vietnam',
    'Bolivia, Plurinational State of': 'Bolivia',
    "Côte d'Ivoire": "Cote d'Ivoire",
    'Türkiye': 'Turkey',
    'Congo, The Democratic Republic of the': 'Congo, Dem. Rep.',
    "Lao People's Democratic Republic": 'Laos',
    'Iran, Islamic Republic of': 'Iran',
    'Venezuela, Bolivarian Republic of': 'Venezuela',
    "Korea, Democratic People's Republic of": "Korea, Dem. People's Rep.",
    'Korea, Republic of': 'Korea',
    'Slovakia': 'Slovak Republic',
}
trade_filtered['Donor'] = trade_filtered['Donor'].replace(PYCOUNTRY_TO_CRDF)
trade_filtered['Recipient'] = trade_filtered['Recipient'].replace(PYCOUNTRY_TO_CRDF)
trade_filtered['refYear'] = pd.to_numeric(trade_filtered['TIME_PERIOD'], errors='coerce')

# Unbekannte Länder entfernen und ausgeben
trade_filtered.dropna(subset=['Donor', 'Recipient', 'refYear'], inplace=True)
nan_donors = trade_filtered[trade_filtered['Donor'].isna()]['REF_AREA'].unique()
if len(nan_donors) > 0:
    print(f"Nicht erkannte Donor-ISO-Codes (werden entfernt): {list(nan_donors)}")

# Pivot: Exporte und Importe in separate Spalten
trade_flow = trade_filtered.pivot_table(
    index=['Donor', 'Recipient', 'refYear'],
    columns='TRADE_FLOW',
    values='OBS_VALUE_USD',
    fill_value=0
).reset_index()

# Spaltennamen bereinigen
trade_flow.columns = [str(col) for col in trade_flow.columns]
rename_map = {}
if 'X' in trade_flow.columns:
    rename_map['X'] = 'Export'
if 'M' in trade_flow.columns:
    rename_map['M'] = 'Import'
trade_flow.rename(columns=rename_map, inplace=True)

# Trade Volume berechnen
trade_flow['Trade_Volume'] = trade_flow.get('Export', 0) + trade_flow.get('Import', 0)

print(f"Bilaterale Handelsdaten (Exports + Imports) nach Aggregation: {len(trade_flow)} Zeilen")
print(trade_flow.head())

# ------------------- 2b. Trade Volume (beide Exportrichtungen) -------------------
# Exporte des Recipienten an den Donor holen (Spiegelrichtung)
trade_flow_reverse = trade_flow.rename(
    columns={'Donor': 'Recipient', 'Recipient': 'Donor', 'Export': 'Reverse_Export'}
)

# Zusammenführen: Für jedes (Donor, Recipient, refYear) die Reverse-Exporte dazuholen
trade_flow = trade_flow.merge(
    trade_flow_reverse[['Donor', 'Recipient', 'refYear', 'Reverse_Export']],
    on=['Donor', 'Recipient', 'refYear'],
    how='left'
)

# Fehlende Reverse-Exporte = 0 (wenn keine Exporte in die andere Richtung)
trade_flow['Reverse_Export'] = trade_flow['Reverse_Export'].fillna(0)

# Trade Volume = Export (Donor→Recipient) + Reverse_Export (Recipient→Donor)
trade_flow['Trade_Volume'] = trade_flow['Export'] + trade_flow['Reverse_Export']

# Aufräumen
trade_flow.drop(columns=['Reverse_Export'], inplace=True)
print(f"Trade Volume (beide Richtungen) berechnet: {len(trade_flow)} Zeilen")
print(trade_flow[['Donor', 'Recipient', 'refYear', 'Export', 'Trade_Volume']].head())

# ------------------- 3. CRDF bilateral -------------------
crdf = crdf_df.copy()
crdf['Amount'] = pd.to_numeric(
    crdf['Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand'],
    errors='coerce'
)
crdf = crdf[crdf['Amount'] > 0]
crdf.rename(columns={
    'Provider': 'Donor',
    'Recipient': 'Recipient',
    'Year': 'refYear',
    'Amount': 'CBA_Commitment'
}, inplace=True)

cba_vuln = crdf[crdf['Recipient'].isin(vuln_countries)].copy()
excluded_recipients = crdf[~crdf['Recipient'].isin(vuln_countries)]['Recipient'].unique()
print(f"Empfänger aus CRDF ausgeschlossen (nicht in Vulnerability): {list(excluded_recipients)}")

cba_bilateral = cba_vuln.groupby(['Donor', 'Recipient', 'refYear'], as_index=False).agg(
    {'CBA_Commitment': 'sum'}
)
print(f"CRDF bilaterale Zeilen (nur Länder als Empfänger): {len(cba_bilateral)}")

# ------------------- 4. Merge -------------------
merged = trade_flow.merge(
    cba_bilateral,
    on=['Donor', 'Recipient', 'refYear'],
    how='inner'
)
merged = merged.dropna(subset=['CBA_Commitment', 'Trade_Volume'])
print(f"\nBeobachtungen nach Merge: {len(merged)}")
print(merged[['Donor', 'Recipient', 'refYear', 'CBA_Commitment', 'Trade_Volume']].head())

# ------------------- 5. Vulnerability dynamisch pro Jahr -------------------
merged.rename(columns={'Recipient': 'Country'}, inplace=True)

merged = merged.merge(
    vuln_long,
    on=['Country', 'refYear'],
    how='left'
)
nan_vuln = merged[merged['Vulnerability_Year'].isna()]['Country'].unique()
if len(nan_vuln) > 0:
    print(f"Empfänger ohne jährlichen Vulnerability-Wert (werden entfernt): {list(nan_vuln)}")

merged = merged.dropna(subset=['Vulnerability_Year'])
print(f"Nach Entfernen von NaN in Vulnerability: {len(merged)} Zeilen")

# ------------------- 6. Log-Transformationen -------------------
merged['log_CBA'] = np.log1p(merged['CBA_Commitment'])
merged['log_Trade_Volume'] = np.log1p(merged['Trade_Volume'])

# ------------------- 7. Deskriptive Statistiken -------------------
if len(merged) >= 2:
    print("\nDeskriptive Statistiken (original + log):")
    print(merged[['CBA_Commitment', 'Trade_Volume', 'Vulnerability_Year',
                  'log_CBA', 'log_Trade_Volume']].describe())

    # ------------------- 8. Korrelationen -------------------
    print("\n--- Pearson Korrelationen ---")
    r1, p1 = pearsonr(merged['CBA_Commitment'], merged['Trade_Volume'])
    print(f"Pearson r(CBA, Trade_Volume) = {r1:.4f}, p = {p1:.4f}")

    r2, p2 = pearsonr(merged['log_CBA'], merged['log_Trade_Volume'])
    print(f"Pearson r(log(CBA), log(Trade_Volume)) = {r2:.4f}, p = {p2:.4f}")

    r3, p3 = pearsonr(merged['CBA_Commitment'], merged['Vulnerability_Year'])
    print(f"Pearson r(CBA, Vulnerability_Year) = {r3:.4f}, p = {p3:.4f}")

    r4, p4 = pearsonr(merged['log_CBA'], merged['Vulnerability_Year'])
    print(f"Pearson r(log(CBA), Vulnerability_Year) = {r4:.4f}, p = {p4:.4f}")

    # ------------------- 9. OLS Regression (log-log) -------------------
    X_log = merged[['log_Trade_Volume', 'Vulnerability_Year']]
    X_log = sm.add_constant(X_log)
    y_log = merged['log_CBA']
    model_log = sm.OLS(y_log, X_log).fit()
    print("\n--- OLS: log(CBA) ~ log(Trade_Volume) + Vulnerability_Year ---")
    print(model_log.summary())

# ... (vorheriger Code unverändert, bis Modell 9 OLS)

# ------------------- 11. Clustered Standard Errors (Donor-Ebene) ----------
print("\n" + "=" * 70)
print("11. OLS MIT CLUSTERED STANDARD ERRORS (CLUSTER = Donor)")
print("=" * 70)
print(f"Warum: Durbin-Watson={durbin_watson(model_log.resid):.3f} zeigt Autokorrelation -> Standard-SE unterschätzt Varianz.\n")

model_clustered = sm.OLS(y_log, X_log).fit(
    cov_type='cluster',
    cov_kwds={'groups': merged['Donor']}
)
print(model_clustered.summary())

# Vergleich: Wie stark ändern sich die SE?
print("\n--- SE-Vergleich: Standard vs. Clustered ---")
se_ratio = model_clustered.bse / model_log.bse
for var in X_log.columns:
    print(f"  {var}: SE Standard={model_log.bse[var]:.4f} | "
          f"Clustered={model_clustered.bse[var]:.4f} | "
          f"Faktor={se_ratio[var]:.2f}x")

# ------------------- 12. Donor Fixed Effects (via Dummies) ---------------
print("\n" + "=" * 70)
print("12. DONOR FIXED EFFECTS (via Dummies) + Clustered SE")
print("=" * 70)
print("Warum: Kontrolliert für zeitinvariante Gebercharakteristika\n")

# Donor-Dummies erstellen (ohne NA)
merged_clean = merged.dropna(subset=['Donor']).copy()
donor_dummies = pd.get_dummies(merged_clean['Donor'], prefix='D', drop_first=True)
donor_dummies = donor_dummies.astype(int)

X3 = pd.concat([
    sm.add_constant(merged_clean[['log_Trade_Volume', 'Vulnerability_Year']]),
    donor_dummies
], axis=1)
X3 = X3.apply(pd.to_numeric, errors='coerce')
y3 = merged_clean['log_CBA']

model_donor_fe = sm.OLS(y3, X3).fit(
    cov_type='cluster',
    cov_kwds={'groups': merged_clean['Donor']}
)

print(f"R²: {model_donor_fe.rsquared:.4f} | Adj. R²: {model_donor_fe.rsquared_adj:.4f}")
print(f"N Beobachtungen: {model_donor_fe.nobs:.0f} | N Donor-Dummies: {donor_dummies.shape[1]}")

# F-Test auf Poolability: Donor-FE-Modell vs. gepooltes OLS (beide mit Standard-SE geschätzt)
fe_plain = sm.OLS(y3, X3).fit()
pooled_plain = sm.OLS(y3, X3[['const', 'log_Trade_Volume', 'Vulnerability_Year']]).fit()
f_pool, p_pool, df_pool = fe_plain.compare_f_test(pooled_plain)
print(f"F-Test Poolability (Donor FE vs. pooled): F={f_pool:.2f}, df=({df_pool:.0f}, {fe_plain.df_resid:.0f}), p={p_pool:.2e}")
print("\nHauptkoeffizienten (ohne Dummies):")
main_vars = ['const', 'log_Trade_Volume', 'Vulnerability_Year']
for var in main_vars:
    coef = model_donor_fe.params[var]
    se   = model_donor_fe.bse[var]
    t    = model_donor_fe.tvalues[var]
    p    = model_donor_fe.pvalues[var]
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {var:25s}: coef={coef:8.4f}, SE={se:.4f}, t={t:6.2f}, p={p:.4f} {sig}")

# ------------------- 13. Panel-Regression mit linearmodels ----------------
try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("\nWARNUNG: linearmodels nicht installiert. Installiere mit: pip install linearmodels")

if HAS_LINEARMODELS:
    print("\n" + "=" * 70)
    print("13. PANEL OLS – Donor + Year Fixed Effects (linearmodels)")
    print("=" * 70)
    print("Warum: Gleichzeitige Kontrolle für Gebereffekte + Zeittrends\n")

    # Panel-Index: entity = Donor, time = refYear
    panel_data = merged.dropna(subset=['log_Trade_Volume', 'Vulnerability_Year', 'log_CBA'])
    panel_data = panel_data.set_index(['Donor', 'refYear']).sort_index()

    y_panel = panel_data['log_CBA']
    X_panel = panel_data[['log_Trade_Volume', 'Vulnerability_Year']]

    model_panel = PanelOLS(
        y_panel, X_panel,
        entity_effects=True,  # Donor-FE
        time_effects=True,    # Year-FE
        drop_absorbed=True
    ).fit(cov_type='clustered', cluster_entity=True)

    print(model_panel.summary)

    print("\n--- Panel-Koeffizienten (nur Prädiktoren) ---")
    for var in ['log_Trade_Volume', 'Vulnerability_Year']:
        coef = model_panel.params[var]
        se   = model_panel.std_errors[var]
        t    = model_panel.tstats[var]
        p    = model_panel.pvalues[var]
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:25s}: coef={coef:8.4f}, SE={se:.4f}, t={t:6.2f}, p={p:.4f} {sig}")

# ------------------- 14. Ausreißeranalyse (Cook's Distance) --------------
print("\n" + "=" * 70)
print("14. AUSREISSER-DIAGNOSTIK (Cook's Distance auf Baseline-OLS)")
print("=" * 70)

from statsmodels.stats.outliers_influence import OLSInfluence

influence  = OLSInfluence(model_log)
cooks_d    = influence.cooks_distance[0]
threshold  = 4 / len(merged)
outlier_mask = cooks_d > threshold
n_outliers = outlier_mask.sum()

print(f"Schwellenwert (4/N = 4/{len(merged)}): {threshold:.5f}")
print(f"Einflussreiche Beobachtungen: {n_outliers} ({n_outliers/len(merged)*100:.1f}%)")

top_outliers = merged[outlier_mask].copy()
if len(top_outliers) > 0:
    top_outliers['cooks_d'] = cooks_d[outlier_mask]
    top_outliers_sorted = top_outliers.sort_values('cooks_d', ascending=False).head(15)
    print("\nTop-15 einflussreichste Beobachtungen:")
    print(top_outliers_sorted[['Donor', 'Country', 'refYear',
                                'CBA_Commitment', 'Trade_Volume',
                                'Vulnerability_Year', 'cooks_d']].to_string(index=False))

# Robustheitscheck: Ohne Ausreißer
print("\n--- Robustheitscheck: OLS ohne Ausreißer ---")
merged_no_outliers = merged[~outlier_mask].copy()
X_no = sm.add_constant(merged_no_outliers[['log_Trade_Volume', 'Vulnerability_Year']])
y_no = merged_no_outliers['log_CBA']
model_no_outliers = sm.OLS(y_no, X_no).fit(
    cov_type='cluster',
    cov_kwds={'groups': merged_no_outliers['Donor']}
)
print(f"N nach Entfernen: {len(merged_no_outliers)} (entfernt: {n_outliers})")
for var in ['const', 'log_Trade_Volume', 'Vulnerability_Year']:
    orig = model_clustered.params[var]
    new  = model_no_outliers.params[var]
    delta = ((new - orig) / abs(orig)) * 100 if orig != 0 else float('nan')
    print(f"  {var:25s}: mit Ausreißern={orig:.4f} | ohne={new:.4f} | Δ={delta:+.1f}%")

# ------------------- 15. Modellvergleich -------------------
print("\n" + "=" * 70)
print("15. MODELLVERGLEICH – Übersicht")
print("=" * 70)

models = {
    "M1: OLS Baseline":           model_log,
    "M2: OLS Clustered SE":       model_clustered,
    "M3: Donor FE + Clust.":      model_donor_fe,
    "M4: No Outliers + Clust.":   model_no_outliers,
}
if HAS_LINEARMODELS:
    models["M5: Panel FE (Donor+Year)"] = model_panel

header = f"{'Modell':<30} {'R²':>6} {'N':>5} {'β log_Trade':>14} {'β Vulnerability':>16}"
print(header)
print("-" * len(header))
for name, m in models.items():
    b_trade = m.params.get('log_Trade_Volume', float('nan'))
    b_vuln  = m.params.get('Vulnerability_Year', float('nan'))
    nobs    = int(m.nobs)
    print(f"{name:<30} {m.rsquared:>6.4f} {nobs:>5} {b_trade:>14.4f} {b_vuln:>16.4f}")

