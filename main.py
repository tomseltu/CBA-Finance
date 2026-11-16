from preprocessing import load_all_datasets
from plots import create_all_plots
import os

# Output Ordner erstellen
if not os.path.exists('output'):
    os.makedirs('output')

# Pfade definieren
vulnerability_path = "data/vulnerability.csv"
population_path = "data/population.csv"
crdf_path = "data/CRDF-RP.csv"

# Datensätze laden
vulnerability_df, population_df, crdf_df = load_all_datasets(
    vulnerability_path, population_path, crdf_path
)
print("Datensatz Größe: " + str(len(crdf_df)))

# CSV Export des gefilterten Datensatzes
csv_output_path = "output/filtered_crdf_data.csv"
crdf_df.to_csv(csv_output_path, index=False, encoding='utf-8')
print(f"Gefilterter Datensatz exportiert: {csv_output_path}")

# Plots erstellen
print("\n=== Erstelle Visualisierungen ===")
plots = create_all_plots(vulnerability_df, population_df, crdf_df, save_html=True)

print("\n=== Fertig! ===")
print("Alle Plots wurden im 'output' Ordner gespeichert!")