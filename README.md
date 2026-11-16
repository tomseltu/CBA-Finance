# Quantifying international climate finance for community-based adaptation

Replication code for the article *Quantifying international climate finance for community-based adaptation projects – mapping of financial commitments, comparison with vulnerability, geopolitical alignment, concessional debt development* (Selje, Polomsky, Nedopil, Heinz, Strikker).

The code identifies adaptation-finance commitments that use language associated with community-based adaptation (CBA) in the OECD DAC recipient-perspective climate finance data (2011–2023). It then produces the figures, the Supplementary Annex D tables, the chi² tests of the keyword dimensions and the trade/vulnerability regressions reported in the article.

## Requirements

- Python 3.13
- Packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data

The raw data are not part of this repository. Download them as described below and place them in `data/` under the following file names:

| File | Source | Used for |
|---|---|---|
| `data/CRDF-RP.csv` | OECD Climate-related development finance, recipient perspective (CRDF-RP) | All analyses |
| `data/vulnerability.csv` | ND-GAIN Country Index, vulnerability scores | Scatter plot, Annex D3, regression |
| `data/population.csv` | World Bank World Development Indicators, population | Per-capita values |
| `data/trade.csv` | OECD Balanced International Merchandise Trade Statistics (BIMTS), HS2017 | Regression |

<!-- TODO: download instructions for the original data -->
```
[Download instructions to be added]
```

Expected file formats (as read by the code):

- `CRDF-RP.csv`: semicolon-separated, Latin-1 encoded, decimal comma, one row per transaction with the original OECD column names. This is the format of a CSV export of the OECD Excel file with a German locale.
- `vulnerability.csv`: ND-GAIN `vulnerability.csv` with the columns `ISO3`, `Name` and one column per year (`1995` … `2023`).
- `population.csv`: World Bank DataBank CSV export with the columns `Country Name` and `2023 [YR2023]`.
- `trade.csv`: OECD Data Explorer CSV export of dataflow `DF_BIMTS_HS2017_2D`. It needs at least the columns `REF_AREA`, `COUNTERPART_AREA`, `TRADE_FLOW`, `PRODUCT_TYPE`, `PRODUCT_HS`, `TIME_PERIOD`, `OBS_VALUE` and `UNIT_MULT`.

## Running the analysis

```bash
python run_all.py
```

This runs all steps in order (about 3 minutes) and writes all results to `output/`. The steps can also be run individually:

| Script | Output |
|---|---|
| `main.py` | `filtered_crdf_data.csv` (all CBA-framed commitments) and figures for CBA-framed commitments |
| `main_comparison.py` | Figures and tables comparing CBA-framed with general adaptation commitments (`fig1_*`, `fig2_*`, `fig4_*`, `comparison_*.csv`) |
| `data_tables.py` | Supplementary Annex D tables (`annex_D1_*`, `annex_D2_*`, `annex_D3_*`) |
| `dimension_tests.py` | Chi² tests of keyword dimensions vs. funder category, region and year (`dimension_tests.csv`) |
| `regression.py` | Pooled OLS, clustered SE, donor fixed effects, outlier check, two-way fixed effects and F-test for poolability (console output, saved to `regression_log.txt` by `run_all.py`) |

### Figures in the article

| Figure | File in `output/` |
|---|---|
| 1 – Time series by recipient continent | `fig1_timeseries_region_comparison.html` (CBA-framed only: `timeseries_plot.html`) |
| 2 – Time series by funder category | `fig2_donor_timeseries_comparison_total.html` (CBA-framed only: `donor_timeseries_plot.html`) |
| 3 – Sankey provider → continent | `sankey_plot.html` |
| 4 – World map by recipient country | `map_plot.html` |
| 5 – CBA-framed finance per capita vs. ND-GAIN vulnerability | `scatter_plot.html` |
| 6 – Financial instruments over time | `financial_instruments_timeseries_plot.html` |
| 7 – Concessional vs. non-concessional over time | `concessionality_timeseries_plot.html` |

## Method in brief

All steps are implemented in `preprocessing.py`.

1. **Period:** commitments from 2011 to 2023, in constant 2023 USD (column *Adaptation-related development finance – Commitment – 2023 USD thousand*).
2. **General adaptation commitments:** transactions with the Rio marker *Adaptation objective = Principal*, the adaptation-related climate components reported by multilateral development banks, and all Green Climate Fund commitments (recoded as principally targeting adaptation, following Savvidou et al., 2021).
3. **CBA-framed commitments:** keyword search in project title and description (`KEYWORDS` in `preprocessing.py`). Keywords are matched as substrings and grouped into four dimensions:
   - A – socio-spatial
   - B – participation
   - C – governance
   - D – collective action

   A commitment is included if it contains at least one keyword from dimension A, or keywords from at least two of the dimensions B, C and D.
4. **Classification:**
   - Recipients are assigned to continents (`CONTINENT_MAPPING`); unspecified multi-country entries are reported as *Unspecified (developing countries)*.
   - Providers are assigned to funder categories (`FUNDER_CATEGORIES`).
   - Country names are harmonised across datasets (`COUNTRY_NAME_MAPPING`).

Notes on specific outputs:

- **Per-capita values** (Figure 5) are cumulative 2011–2023 commitments divided by 2023 population. Vulnerability is the mean ND-GAIN vulnerability score across all available years. Annex D3 reports the same per-capita values as an annual average (divided by 13 years).
- **Regression:** the unit of observation is a donor–recipient–year with positive CBA-framed commitments. Only bilateral donors matched to the trade data are included.
- **Treemap:** small sub-sectors are grouped into *Other Subsectors* using sector-specific thresholds (`group_treemap_subsectors` in `plots.py`).

## License

<!-- TODO: choose a license -->
