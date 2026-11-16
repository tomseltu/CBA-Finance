import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from preprocessing import KEYWORDS
import re

# Farbpalette für alle Plots (aus deinem Treemap-Wunsch)
CUSTOM_PAL = [
    "#81aa80", "#747fad", "#8473a9", "#636363", "#b97369",
    "#838e63", "#95b7dd", "#bfd46f", "#a59152"
]

def create_scatter_vulnerability_finance(vulnerability_df, crdf_df, population_df):
    crdf_filtered = crdf_df[crdf_df['Recipient'] != 'REGIONAL'].copy()
    finance_per_country = crdf_filtered.groupby('Recipient').agg({
        'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand': 'sum'
    }).reset_index()
    finance_per_country.columns = ['Name', 'Total_Finance']
    finance_per_country['Total_Finance'] = pd.to_numeric(finance_per_country['Total_Finance'], errors='coerce') * 1000
    finance_per_country = finance_per_country[finance_per_country['Total_Finance'] > 0].copy()
    latest_year_col = '2023 [YR2023]'
    population_latest = population_df[['Country Name', latest_year_col]].copy()
    population_latest.columns = ['Name', 'Population']
    population_latest['Population'] = pd.to_numeric(population_latest['Population'], errors='coerce')
    merged = finance_per_country.merge(population_latest, on='Name', how='left')
    merged['Finance_Per_Capita'] = merged['Total_Finance'] / merged['Population']
    merged = merged.merge(vulnerability_df, on='Name', how='left')
    year_columns = [col for col in vulnerability_df.columns if col.isdigit()]
    if year_columns:
        for col in year_columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
        merged['Vulnerability_Index'] = merged[year_columns].mean(axis=1)
    merged = merged.dropna(subset=['Vulnerability_Index', 'Finance_Per_Capita'])

    scatter_color = "#81aa80"

    fig = go.Figure(go.Scatter(
        x=merged['Vulnerability_Index'],
        y=merged['Finance_Per_Capita'],
        mode='markers',
        marker=dict(color=scatter_color, size=9, opacity=0.8, line=dict(width=1, color="#222")),
        text=merged['Name'],
        hovertemplate='<b>%{text}</b><br>Vulnerability Index: %{x:.2f}<br>Finance per Capita: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis=dict(
            title='Average ND-Gain Vulnerability Index',
            title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
            showline=True,
            linecolor="#e5e5e5",
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
            ticks="outside",
            ticklen=8,
            tickcolor="#e5e5e5",
            tickfont=dict(family="DejaVu Sans,Arial", size=13, color="#444"),
        ),
        yaxis=dict(
            title='CBA Finance per Capita (USD per Person, 2023)',
            title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
            showline=True,
            linecolor="#e5e5e5",
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
            ticks="outside",
            ticklen=8,
            tickcolor="#e5e5e5",
            tickfont=dict(family="DejaVu Sans,Arial", size=13, color="#444"),
        ),
        font=dict(family="DejaVu Sans,Arial", color="#222"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=60, r=30, t=30, b=60)
    )
    return fig

def create_map_global_finance(crdf_df):
    map_data = crdf_df.groupby('Recipient').agg({
        'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand': 'sum'
    }).reset_index()
    map_data.columns = ['Recipient', 'Total_Finance']
    map_data['Total_Finance'] = pd.to_numeric(map_data['Total_Finance'], errors='coerce') / 1000
    map_data = map_data[map_data['Total_Finance'] > 0].copy()

    # Neuer, kontrastreicherer Grünverlauf
    green_gradient = [
        [0.0, "#f8ffe8"],   # sehr helles Grün (fast weiß)
        [0.15, "#bfd46f"],  # helles Grün
        [0.5, "#81aa80"],   # mittleres Grün
        [0.85, "#4b6b3c"],  # dunkleres Grün
        [1.0, "#1b2e13"]    # sehr dunkles Grün
    ]

    fig = px.choropleth(
        map_data,
        locations='Recipient',
        locationmode='country names',
        color='Total_Finance',
        hover_name='Recipient',
        color_continuous_scale=green_gradient,
        labels={'Total_Finance': 'Total Finance (Million USD, 2023)'}
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            bgcolor="white",
            landcolor="white",
            lakecolor="white",
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='DejaVu Sans,Arial', color="#222"),
        margin=dict(l=40, r=40, t=40, b=40),
        coloraxis_colorbar=dict(
            title=dict(
                text="Million USD",
                font=dict(color="#222", family='DejaVu Sans,Arial')
            ),
            tickfont=dict(color="#222", family='DejaVu Sans,Arial'),
            thickness=18,
            len=0.5,
            outlinewidth=0,
            x=0.98,
            y=0.5
        ),
        height=700
    )
    return fig

def create_sankey_funders_continents(crdf_df):
    sankey_data = crdf_df.groupby(['Funder Category', 'Region']).agg({
        'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand': 'sum'
    }).reset_index()
    sankey_data.columns = ['Source', 'Target', 'Value']
    sankey_data['Value'] = pd.to_numeric(sankey_data['Value'], errors='coerce') / 1000
    sankey_data = sankey_data[sankey_data['Value'] > 0].copy().dropna()
    sources = sankey_data['Source'].unique().tolist()
    targets = sankey_data['Target'].unique().tolist()
    left_labels = [f"{s}\u2002\u2002\u2002" for s in sources]
    right_labels = [f"\u2002\u2002\u2002{t}" for t in targets]
    all_labels = left_labels + right_labels
    label_map = {s: i for i, s in enumerate(sources)}
    label_map.update({t: i+len(sources) for i, t in enumerate(targets)})
    sankey_data['source_idx'] = sankey_data['Source'].map(label_map)
    sankey_data['target_idx'] = sankey_data['Target'].map(label_map)
    color_map = {
        "BILATERAL FUNDERS": CUSTOM_PAL[0],
        "PRIVATE FUNDER": CUSTOM_PAL[5],
        "MULTILATERAL DEVELOPMENT BANKS": CUSTOM_PAL[1],
        "CLIMATE FUNDS": CUSTOM_PAL[2],
        "OTHER MULTILATERAL INSTITUTIONS": CUSTOM_PAL[3],
        "FOUNDATIONS / PRIVATE DONORS": CUSTOM_PAL[4],
    }
    link_colors = [color_map.get(source, "#888888") for source in sankey_data['Source']]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=12,
            line=dict(color="#222", width=1),
            label=left_labels + right_labels,
            color="black"
        ),
        link=dict(
            source=sankey_data['source_idx'],
            target=sankey_data['target_idx'],
            value=sankey_data['Value'],
            color=link_colors
        )
    )])
    fig.update_layout(
        font=dict(family='DejaVu Sans,Arial', size=12, color="#222"),
        paper_bgcolor='white',
        height=600,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def create_timeseries_finance(crdf_df):
    crdf_filtered = crdf_df[(crdf_df['Recipient'] != 'REGIONAL') & (crdf_df['Region'].notna())].copy()
    timeseries_data = crdf_filtered.groupby(['Year', 'Region']).agg({
        'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand': 'sum'
    }).reset_index()
    timeseries_data.columns = ['Year', 'Region', 'Total_Finance']
    timeseries_data['Year'] = timeseries_data['Year'].astype(int)
    timeseries_data['Total_Finance'] = timeseries_data['Total_Finance'] / 1000
    timeseries_data = timeseries_data[timeseries_data['Total_Finance'] > 0].copy().dropna()

    fig = go.Figure()
    regions = timeseries_data['Region'].unique()
    for i, region in enumerate(regions):
        region_data = timeseries_data[timeseries_data['Region'] == region]
        fig.add_trace(go.Scatter(
            x=region_data['Year'],
            y=region_data['Total_Finance'],
            mode='lines+markers',
            name=region,
            line=dict(color=CUSTOM_PAL[i % len(CUSTOM_PAL)], width=3),
            marker=dict(size=7, color=CUSTOM_PAL[i % len(CUSTOM_PAL)]),
            showlegend=False
        ))
        last_x = region_data['Year'].max()
        last_y = region_data[region_data['Year'] == last_x]['Total_Finance'].values[0]
        fig.add_annotation(
            x=last_x + 0.2, y=last_y,
            text=region,
            font=dict(family="DejaVu Sans,Arial", size=13, color=CUSTOM_PAL[i % len(CUSTOM_PAL)]),
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="middle"
        )

    fig.update_xaxes(
        tickvals=[x for x in range(timeseries_data['Year'].min(), timeseries_data['Year'].max() + 1, 2)],
        ticktext=[str(x) for x in range(timeseries_data['Year'].min(), timeseries_data['Year'].max() + 1, 2)],
        tickfont=dict(family="DejaVu Sans,Arial", size=13, color="#444"),
        ticks="outside",
        ticklen=12,
        tickcolor="#e5e5e5",
        showline=True,
        linecolor="#e5e5e5",
        showgrid=False
    )
    fig.update_yaxes(
        tickfont=dict(family="DejaVu Sans,Arial", size=11, color="#444"),
        ticks="outside",
        ticklen=8,
        tickcolor="#e5e5e5",
        showline=True,
        linecolor="#e5e5e5",
        showgrid=True,
        gridcolor="#f0f0f0",
        range=[0, None] 
    )
    fig.update_layout(
        margin=dict(l=60, r=30, t=60, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DejaVu Sans,Arial", color="#222"),
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="CBA Finance (Million USD)", title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"))
    fig.update_xaxes(title_text="Year", title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"))
    return fig

def group_treemap_subsectors(df):
    thresholds = {
        "III.1. Agriculture, Forestry, Fishing": "Forestry development",
        "I.4. Water Supply & Sanitation": "River basins development",
        "IV.2. Other Multisector": "Multisector aid",
        "IV.1. General Environment Protection": "Environmental education/training",
        "II.1. Transport & Storage": "Transport policy and administrative management",
        "II.3. Energy": "Energy generation, renewable sources - multiple technologies",
        "I.6. Other Social Infrastructure & Services": "Low-cost housing",
        "I.5. Government & Civil Society": "Public finance management (PFM)",
    }
    full_other_sector = "VIII.2. Reconstruction Relief & Rehabilitation"

    df = df.copy()
    new_rows = []
    drop_idx = []

    # Ganze Kategorie zu Other Sectors
    mask_full = df['Sector'] == full_other_sector
    if mask_full.any():
        sum_val = df.loc[mask_full, 'Total_Finance'].sum()
        new_rows.append({'Sector': 'Other Sectors', 'Sub-sector': full_other_sector, 'Total_Finance': sum_val})
        drop_idx.extend(df[mask_full].index.tolist())

    # Sektoren mit Schwellenwert
    for sector, threshold in thresholds.items():
        mask_sector = df['Sector'] == sector
        sector_df = df[mask_sector].copy()
        if not sector_df.empty:
            try:
                threshold_value = sector_df.loc[sector_df['Sub-sector'] == threshold, 'Total_Finance'].iloc[0]
            except IndexError:
                print(f"[DEBUG] Schwelle '{threshold}' NICHT gefunden in Sektor '{sector}'. Verfügbare Sub-Sektoren: {sector_df['Sub-sector'].tolist()}")
                continue
            # Alle Sub-Sektoren mit Wert <= Schwellenwert zu Other Subsectors
            mask_other = sector_df['Total_Finance'] <= threshold_value
            to_other = sector_df[mask_other]
            if not to_other.empty:
                sum_val = to_other['Total_Finance'].sum()
                new_rows.append({'Sector': sector, 'Sub-sector': 'Other Subsectors', 'Total_Finance': sum_val})
                drop_idx.extend(to_other.index.tolist())

    # Sektoren ohne Schwellenwert komplett zu Other Sectors
    sectors_with_threshold = set(thresholds.keys())
    all_sectors = set(df['Sector'].unique())
    sectors_without_threshold = all_sectors - sectors_with_threshold - {full_other_sector}

    # --- NEU: Sektoren ab "I. 1. Education" zu "Other Sectors"->"Other" zusammenfassen ---
    # Sortiere Sektoren alphabetisch und finde Index von "I. 1. Education"
    sorted_sectors = sorted(sectors_without_threshold)
    if "I. 1. Education" in sorted_sectors:
        idx = sorted_sectors.index("I. 1. Education")
        sectors_for_other = sorted_sectors[idx:]  # alle ab "I. 1. Education"
        mask_other = df['Sector'].isin(sectors_for_other)
        if mask_other.any():
            sum_val = df.loc[mask_other, 'Total_Finance'].sum()
            new_rows.append({'Sector': 'Other Sectors', 'Sub-sector': 'Other', 'Total_Finance': sum_val})
            drop_idx.extend(df[mask_other].index.tolist())
        # Die übrigen Sektoren ohne Schwellenwert wie gehabt behandeln
        sectors_without_threshold = set(sorted_sectors[:idx])
    # ----------------------------------------------------------------------

    for sector in sectors_without_threshold:
        mask_sector = df['Sector'] == sector
        if mask_sector.any():
            sum_val = df.loc[mask_sector, 'Total_Finance'].sum()
            new_rows.append({'Sector': 'Other Sectors', 'Sub-sector': sector, 'Total_Finance': sum_val})
            drop_idx.extend(df[mask_sector].index.tolist())

    df = df.drop(index=drop_idx)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df

def wrap_label(label, maxlen=18):
    import textwrap
    return "<br>".join(textwrap.wrap(str(label), maxlen))

def create_treemap_sector_finance(crdf_df):
    treemap_palette = CUSTOM_PAL
    treemap_data = crdf_df.groupby(['Sector', 'Sub-sector']).agg({
        'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand': 'sum'
    }).reset_index()
    treemap_data.columns = ['Sector', 'Sub-sector', 'Total_Finance']
    treemap_data['Total_Finance'] = pd.to_numeric(treemap_data['Total_Finance'], errors='coerce') / 1000
    treemap_data = treemap_data[treemap_data['Total_Finance'] > 0].copy()
    treemap_data = treemap_data.dropna()
    treemap_data = group_treemap_subsectors(treemap_data)

    # Berechne Gesamtfinanzvolumen pro Sektor und Prozentanteil
    sector_totals = treemap_data.groupby('Sector')['Total_Finance'].sum()
    total_sum = treemap_data['Total_Finance'].sum()
    sector_percent = (sector_totals / total_sum * 100).round(1)

    fig = px.treemap(
        treemap_data,
        path=['Sector', 'Sub-sector'],
        values='Total_Finance',
        color='Sector',
        color_discrete_sequence=treemap_palette,
        hover_data={'Total_Finance': ':,.2f'}
    )
    fig.update_layout(
        uniformtext=dict(minsize=9, mode='hide'), 
        height=600,
        font=dict(family='DejaVu Sans,Arial', color="#222"),
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Labels anpassen: Sektoren mit Zahl und Prozent, Subsektoren mit Zeilenumbruch
    labels = fig.data[0]['labels']
    ids = fig.data[0]['ids']
    new_labels = []
    for label, id_ in zip(labels, ids):
        if '/' in id_:
            # Subsector-Ebene: umbrechen
            new_labels.append(wrap_label(label, maxlen=18))
        else:
            # Sector-Ebene: Zahl und Prozent ergänzen
            sector_value = sector_totals.get(label, 0)
            percent_value = sector_percent.get(label, 0)
            label_new = f"{label}\n{sector_value:,.1f} Mio USD ({percent_value:.1f}%)"
            new_labels.append(label_new)
    fig.data[0]['labels'] = new_labels

    return fig


def create_keyword_wordcloud(crdf_df):
    # Flache Liste aller Suchbegriffe (wie in KEYWORDS definiert)
    all_keywords = []
    for keyword_list in KEYWORDS.values():
        all_keywords.extend(keyword_list)

    # Zähle, wie oft jeder Suchbegriff in den gefundenen Wörtern vorkommt
    word_counts = {k: 0 for k in all_keywords}
    for word_list in crdf_df['Found_Original_Words']:
        if isinstance(word_list, list):
            for found_word in word_list:
                for keyword in all_keywords:
                    # Substring-Match (wie in der Extraktion)
                    if keyword.lower() in found_word.lower():
                        word_counts[keyword] += 1

    # Entferne Suchbegriffe mit Count 0
    word_counts = {k: v for k, v in word_counts.items() if v > 0}

    import random
    rng = random.Random(42)  # fester Seed für reproduzierbare Farben
    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return rng.choice(CUSTOM_PAL)

    if word_counts:
        try:
            wordcloud = WordCloud(
                random_state=42,
                width=800,
                height=400,
                background_color='white',
                max_words=50,
                min_font_size=10,
                max_font_size=80,
                relative_scaling=0.5,
                margin=20,
                font_path=None,
                color_func=custom_color_func
            ).generate_from_frequencies(word_counts)
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_base64}",
                    xref="x", yref="y",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    sizing="stretch",
                    opacity=1,
                    layer="below"
                )
            )
            fig.update_layout(
                xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
                yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
                width=800,
                height=500,
                showlegend=False,
                font=dict(family='DejaVu Sans', color="#222"),
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"WordCloud Fehler: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, family='DejaVu Sans')
            )
            fig.update_layout(width=800, height=500, font=dict(family='DejaVu Sans'))
            return fig
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="Keine Wörter gefunden",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, family='DejaVu Sans')
        )
        fig.update_layout(width=800, height=500, font=dict(family='DejaVu Sans'))
        return fig
    
def group_small_categories(df, col, value_col, threshold=0.03):
    total = df[value_col].sum()
    df['share'] = df[value_col] / total
    large = df[df['share'] >= threshold].copy()
    small = df[df['share'] < threshold].copy()
    if not small.empty:
        other_row = {
            col: 'Other',
            value_col: small[value_col].sum(),
            'share': small['share'].sum()
        }
        large = pd.concat([large, pd.DataFrame([other_row])], ignore_index=True)
    return large.drop(columns='share')

def create_timeseries_financial_instrument(crdf_df):
    df = crdf_df.copy()
    df = df[df['Financial Instrument'].notna()]
    df['Amount'] = pd.to_numeric(df['Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand'], errors='coerce')
    df = df[df['Amount'] > 0]
    grouped = df.groupby(['Year', 'Financial Instrument']).agg({'Amount': 'sum'}).reset_index()
    result = []
    for year, group in grouped.groupby('Year'):
        grouped_year = group.copy()
        grouped_year = group_small_categories(grouped_year, 'Financial Instrument', 'Amount', threshold=0.03)
        grouped_year['Year'] = year
        result.append(grouped_year)
    plot_df = pd.concat(result, ignore_index=True)
    pivot = plot_df.pivot_table(index='Year', columns='Financial Instrument', values='Amount', fill_value=0)
    pivot = pivot.sort_index()
    cats = list(pivot.columns)
    colors = {cat: CUSTOM_PAL[i % len(CUSTOM_PAL)] for i, cat in enumerate(cats)}
    if 'Other' in cats:
        colors['Other'] = "#cccccc"
    fig = go.Figure()
    for cat in cats:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[cat],
            mode='lines',
            stackgroup='one',
            name=cat,
            line=dict(width=0.5, color=colors[cat]),
            hoverinfo='x+y+name',
            showlegend=True
        ))
    fig.update_layout(
        font=dict(family='DejaVu Sans,Arial', color="#222"),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=30, t=30, b=60),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(245,245,245,0.8)',
            bordercolor="#eee",
            borderwidth=1,
            font=dict(family='DejaVu Sans,Arial', color="#222")
        ),
        hovermode='x unified'
    )
    fig.update_xaxes(
        title_text="Year",
        title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
        tickfont=dict(family="DejaVu Sans,Arial", size=13, color="#444"),
        showline=True,
        linecolor="#e5e5e5",
        showgrid=False
    )
    fig.update_yaxes(
        title_text="CBA Finance (Thousand USD)",
        title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
        tickfont=dict(family="DejaVu Sans,Arial", size=11, color="#444"),
        showline=True,
        linecolor="#e5e5e5",
        showgrid=True,
        gridcolor="#f0f0f0"
    )
    return fig

def create_timeseries_concessionality(crdf_df):
    df = crdf_df.copy()
    df = df[df['Concessionality'].notna()]
    df['Amount'] = pd.to_numeric(df['Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand'], errors='coerce')
    df = df[df['Amount'] > 0]
    grouped = df.groupby(['Year', 'Concessionality']).agg({'Amount': 'sum'}).reset_index()
    result = []
    for year, group in grouped.groupby('Year'):
        grouped_year = group.copy()
        grouped_year = group_small_categories(grouped_year, 'Concessionality', 'Amount', threshold=0.03)
        grouped_year['Year'] = year
        result.append(grouped_year)
    plot_df = pd.concat(result, ignore_index=True)
    pivot = plot_df.pivot_table(index='Year', columns='Concessionality', values='Amount', fill_value=0)
    pivot = pivot.sort_index()
    cats = list(pivot.columns)
    colors = {cat: CUSTOM_PAL[i % len(CUSTOM_PAL)] for i, cat in enumerate(cats)}
    if 'Other' in cats:
        colors['Other'] = "#cccccc"
    fig = go.Figure()
    for cat in cats:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[cat],
            mode='lines',
            stackgroup='one',
            name=cat,
            line=dict(width=0.5, color=colors[cat]),
            hoverinfo='x+y+name',
            showlegend=True
        ))
    fig.update_layout(
        font=dict(family='DejaVu Sans,Arial', color="#222"),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=30, t=30, b=60),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(245,245,245,0.8)',
            bordercolor="#eee",
            borderwidth=1,
            font=dict(family='DejaVu Sans,Arial', color="#222")
        ),
        hovermode='x unified'
    )
    fig.update_xaxes(
        title_text="Year",
        title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
        tickfont=dict(family="DejaVu Sans,Arial", size=13, color="#444"),
        showline=True,
        linecolor="#e5e5e5",
        showgrid=False
    )
    fig.update_yaxes(
        title_text="CBA Finance (Thousand USD)",
        title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
        tickfont=dict(family="DejaVu Sans,Arial", size=11, color="#444"),
        showline=True,
        linecolor="#e5e5e5",
        showgrid=True,
        gridcolor="#f0f0f0"
    )
    return fig

def create_timeseries_donor_comparison(crdf_df):
    """
    Zeitreihe der CBA Finance (Commitments) gruppiert nach Funder Category.
    Zeigt, wie sich die Beiträge der verschiedenen Geberkategorien über die Zeit entwickeln.
    """
    df = crdf_df.copy()
    df = df[df['Funder Category'].notna()]
    df['Amount'] = pd.to_numeric(
        df['Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand'],
        errors='coerce'
    )
    df = df[df['Amount'] > 0]

    grouped = df.groupby(['Year', 'Funder Category']).agg({'Amount': 'sum'}).reset_index()

    # Kleine Kategorien pro Jahr zusammenfassen (threshold 3%)
    result = []
    for year, group in grouped.groupby('Year'):
        grouped_year = group.copy()
        grouped_year = group_small_categories(grouped_year, 'Funder Category', 'Amount', threshold=0.03)
        grouped_year['Year'] = year
        result.append(grouped_year)
    plot_df = pd.concat(result, ignore_index=True)

    pivot = plot_df.pivot_table(index='Year', columns='Funder Category', values='Amount', fill_value=0)
    pivot = pivot.sort_index()
    cats = list(pivot.columns)

    colors = {cat: CUSTOM_PAL[i % len(CUSTOM_PAL)] for i, cat in enumerate(cats)}
    if 'Other' in cats:
        colors['Other'] = "#cccccc"

    fig = go.Figure()
    for cat in cats:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[cat],
            mode='lines',
            stackgroup='one',
            name=cat,
            line=dict(width=0.5, color=colors[cat]),
            hoverinfo='x+y+name',
            showlegend=True
        ))

    fig.update_layout(
        title=dict(
            text='CBA Finance Commitments by Funder Category',
            font=dict(family='DejaVu Sans,Arial', size=16, color="#222"),
            x=0.5,
            xanchor='center'
        ),
        font=dict(family='DejaVu Sans,Arial', color="#222"),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(245,245,245,0.8)',
            bordercolor="#eee",
            borderwidth=1,
            font=dict(family='DejaVu Sans,Arial', color="#222")
        ),
        hovermode='x unified'
    )
    fig.update_xaxes(
        title_text="Year",
        title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
        tickfont=dict(family="DejaVu Sans,Arial", size=13, color="#444"),
        showline=True,
        linecolor="#e5e5e5",
        showgrid=False
    )
    fig.update_yaxes(
        title_text="CBA Finance (Thousand USD)",
        title_font=dict(family="DejaVu Sans,Arial", size=14, color="#444"),
        tickfont=dict(family="DejaVu Sans,Arial", size=11, color="#444"),
        showline=True,
        linecolor="#e5e5e5",
        showgrid=True,
        gridcolor="#f0f0f0"
    )
    return fig

def create_financial_instrument_distribution(crdf_df):
    instrument_counts = crdf_df['Financial Instrument'].value_counts().reset_index()
    instrument_counts.columns = ['Financial_Instrument', 'Count']
    instrument_counts = instrument_counts[instrument_counts['Financial_Instrument'].notna()]

    fig = go.Figure(go.Pie(
        labels=instrument_counts['Financial_Instrument'],
        values=instrument_counts['Count'],
        hole=0.5,
        marker=dict(colors=CUSTOM_PAL[:len(instrument_counts)],
                    line=dict(color="#fff", width=1)),
        textfont=dict(color="#222", size=12, family='DejaVu Sans,Arial'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='white',
        font=dict(family='DejaVu Sans,Arial', color="#222"),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(245,245,245,0.8)',
            bordercolor="#eee",
            borderwidth=1,
            font=dict(family='DejaVu Sans,Arial', color="#222")
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def create_concessionality_distribution(crdf_df):
    concessionality_counts = crdf_df['Concessionality'].value_counts().reset_index()
    concessionality_counts.columns = ['Concessionality', 'Count']
    concessionality_counts = concessionality_counts[concessionality_counts['Concessionality'].notna()]

    fig = go.Figure(go.Pie(
        labels=concessionality_counts['Concessionality'],
        values=concessionality_counts['Count'],
        hole=0.5,
        marker=dict(colors=CUSTOM_PAL[:len(concessionality_counts)],
                    line=dict(color="#fff", width=1)),
        textfont=dict(color="#222", size=12, family='DejaVu Sans,Arial'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='white',
        font=dict(family='DejaVu Sans,Arial', color="#222"),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(245,245,245,0.8)',
            bordercolor="#eee",
            borderwidth=1,
            font=dict(family='DejaVu Sans,Arial', color="#222")
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def create_all_plots(vulnerability_df, population_df, crdf_df, save_html=True):
    plots = {}
    print("Erstelle Scatter Plot...")
    plots['scatter'] = create_scatter_vulnerability_finance(vulnerability_df, crdf_df, population_df)
    print("Erstelle Sankey Diagram...")
    plots['sankey'] = create_sankey_funders_continents(crdf_df)
    print("Erstelle Time Series...")
    plots['timeseries'] = create_timeseries_finance(crdf_df)
    print("Erstelle Global Map...")
    plots['map'] = create_map_global_finance(crdf_df)
    print("Erstelle Treemap...")
    plots['treemap'] = create_treemap_sector_finance(crdf_df)
    print("Erstelle Keyword Word Cloud...")
    plots['wordcloud'] = create_keyword_wordcloud(crdf_df)
    print("Erstelle Financial Instrument Distribution...")
    plots['financial_instruments'] = create_financial_instrument_distribution(crdf_df)
    print("Erstelle Concessionality Distribution...")
    plots['concessionality'] = create_concessionality_distribution(crdf_df)
    print("Erstelle Financial Instrument Time Series...")
    plots['financial_instruments_timeseries'] = create_timeseries_financial_instrument(crdf_df)
    print("Erstelle Concessionality Time Series...")
    plots['concessionality_timeseries'] = create_timeseries_concessionality(crdf_df)
    print("Erstelle Donor Time Series...")
    plots['donor_timeseries'] = create_timeseries_donor_comparison(crdf_df)

    if save_html:
        for name, fig in plots.items():
            filename = f'output/{name}_plot.html'
            fig.write_html(filename)
            print(f"Gespeichert: {filename}")
    return plots