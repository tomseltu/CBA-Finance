"""Kategoriale Analyse: Dimensionen A–D vs. Funder/Region/Year.

Methodik:
- Chi²-Unabhängigkeitstest (Pearson), ohne Yates-Korrektur
  (bei 2×N, N>2 ohnehin nicht angewandt).
- Cramér's V, bias-korrigiert nach Bergsma (2013).
- Multiple-Testing-Korrektur: Holm-Bonferroni über alle 12 Tests.
- Pearson-Residuen (O-E)/sqrt(E).
- Ergebnisse werden als CSV exportiert.
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from preprocessing import load_all_datasets

# UTF-8 Ausgabe, sonst Abbruch bei Umleitung in Datei (Windows cp1252)
sys.stdout.reconfigure(encoding='utf-8')

# ---------- 1. Daten laden ----------
BASE = Path(__file__).resolve().parent

vulnerability_path = BASE / "data" / "vulnerability.csv"
population_path    = BASE / "data" / "population.csv"
crdf_path          = BASE / "data" / "CRDF-RP.csv"

print("Lade CRDF-Daten...")
vulnerability_df, population_df, crdf_df = load_all_datasets(
    str(vulnerability_path), str(population_path), str(crdf_path)
)
crdf = crdf_df.copy()

# Year sauber als String (statt float mit .0) -> schön für Paper-Tabellen
if 'Year' in crdf.columns:
    crdf['Year'] = crdf['Year'].dropna().astype(int).reindex(crdf.index).astype('Int64').astype(str)

# ---------- 2. Hilfsfunktionen ----------
def cramers_v_corrected(chi2, n, rows, cols):
    """Bias-korrigiertes Cramér's V (Bergsma 2013)."""
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (cols - 1) * (rows - 1) / (n - 1))
    r_corr = rows - (rows - 1) ** 2 / (n - 1)
    k_corr = cols - (cols - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan

def pearson_residuals(observed, expected):
    return (observed - expected) / np.sqrt(expected)

def run_chi2_test(contingency_table, title="Kreuztabelle"):
    # Leere Zeilen/Spalten entfernen + loggen
    empty_cols = contingency_table.columns[contingency_table.sum(axis=0) == 0].tolist()
    empty_rows = contingency_table.index[contingency_table.sum(axis=1) == 0].tolist()
    if empty_cols or empty_rows:
        print(f"  ⚠ Leere entfernt – Zeilen: {empty_rows}, Spalten: {empty_cols}")

    table = contingency_table.drop(index=empty_rows, columns=empty_cols, errors='ignore')
    if table.size == 0:
        print(f"\n{'='*70}\n{title}\n{'='*70}\nLeere Tabelle – übersprungen.")
        return None

    n = int(table.values.sum())
    rows, cols = table.shape

    # Chi² ohne Yates (bei 2×N, N>2 ohnehin identisch)
    chi2, p, dof, expected = chi2_contingency(table.values, correction=False)

    # Warnung bei kleinen Erwartungswerten
    small_pct = (expected < 5).mean()
    if small_pct > 0.20:
        print(f"  ⚠ {small_pct:.0%} der erwarteten Häufigkeiten < 5 → "
              f"Chi² unzuverlässig (Fisher/Permutation erwägen).")

    cv = cramers_v_corrected(chi2, n, rows, cols)
    resid = pearson_residuals(table.values, expected)

    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"Tabelle ({rows} × {cols}):")
    print(table.to_string())
    print(f"\nChi² = {chi2:.3f}  |  df = {dof}  |  p = {p:.4g}  |  Cramér's V (bc) = {cv:.3f}")
    print(f"Interpretation: {'Signifikanter Zusammenhang (p < 0.05)' if p < 0.05 else 'Kein signifikanter Zusammenhang'}")

    resid_df = pd.DataFrame(resid, index=table.index, columns=table.columns)
    print("\nPearson-Residuen (standardisiert):")
    print(resid_df.round(2).to_string())
    print("\nHinweis: |r| > 1.96 → signifikante Abweichung (α=0.05).")
    return {'chi2': chi2, 'dof': dof, 'p': p, 'cramers_v': cv,
            'residuals': resid_df, 'n': n}

# ---------- 3. Dimensionen-Tests ----------
dimensions = ['Has_Dimension_A', 'Has_Dimension_B', 'Has_Dimension_C', 'Has_Dimension_D']
dim_labels = {
    'Has_Dimension_A': 'Dimension A (local/community)',
    'Has_Dimension_B': 'Dimension B (participatory)',
    'Has_Dimension_C': 'Dimension C (decentralized)',
    'Has_Dimension_D': 'Dimension D (collective/commoning)',
}
variables = [('Funder Category', 'Funder Category'),
             ('Region', 'Region'),
             ('Year', 'Year')]

results = []   # Sammeln für Holm-Korrektur + Export

for dim in dimensions:
    for var, varlabel in variables:
        print(f"\n\n>>> {dim_labels[dim]} × {varlabel}")
        ct = pd.crosstab(crdf[dim], crdf[var], margins=True, margins_name='Total')
        ct = ct.drop(index='Total', columns='Total', errors='ignore')
        res = run_chi2_test(ct, f"{dim_labels[dim]} × {varlabel}")
        if res is not None:
            results.append({
                'dimension': dim_labels[dim],
                'variable': varlabel,
                'chi2': res['chi2'],
                'dof': res['dof'],
                'p_raw': res['p'],
                'cramers_v_bc': res['cramers_v'],
                'n': res['n'],
            })

# ---------- 4. Holm-Korrektur über alle 12 Tests ----------
df_res = pd.DataFrame(results)
if not df_res.empty:
    _, p_holm, _, _ = multipletests(df_res['p_raw'].values, method='holm')
    df_res['p_holm'] = p_holm
    df_res['sig_raw']  = df_res['p_raw']  < 0.05
    df_res['sig_holm'] = df_res['p_holm'] < 0.05

    print("\n\n" + "=" * 90)
    print("ZUSAMMENFASSUNG (Holm-Bonferroni über alle 12 Tests, α = 0.05)")
    print("=" * 90)
    print(df_res[['dimension', 'variable', 'chi2', 'dof',
                  'p_raw', 'p_holm', 'cramers_v_bc',
                  'sig_raw', 'sig_holm']].to_string(index=False))

    out_path = BASE / "output" / "dimension_tests.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df_res.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")

print("\nFertig – Dimensionen-Analyse abgeschlossen.")