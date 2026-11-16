from preprocessing import preprocess_crdf
from plots_comparison import create_all_comparison_plots
import os

# Vergleich CBA-framed vs. General Adaptation Commitments (Figure 1, 2, 4)
if not os.path.exists('output'):
    os.makedirs('output')

crdf_path = "data/CRDF-RP.csv"

# General Adaptation: Principal-Marker (+ GCF manuell Principal), ohne Keyword Filter
general_df = preprocess_crdf(crdf_path, apply_keyword_filter=False)
# CBA-framed: gleiches Datenset mit Keyword Filter
cba_df = preprocess_crdf(crdf_path)
print(f"General Adaptation: {len(general_df)} | CBA-framed: {len(cba_df)}")

create_all_comparison_plots(cba_df, general_df, save_html=True)
