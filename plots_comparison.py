import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plots import CUSTOM_PAL, group_small_categories

FINANCE_COL = 'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand'
FONT = "DejaVu Sans,Arial"
GENERAL_LABEL = "General adaptation commitments"
CBA_LABEL = "CBA-framed commitments"

# Lesbare Funder-Bezeichnungen für Legenden und Panel-Titel
FUNDER_LABELS = {
    'BILATERAL FUNDERS': 'Bilateral funders',
    'MULTILATERAL DEVELOPMENT BANKS': 'Multilateral development banks',
    'CLIMATE FUNDS': 'Climate funds',
    'OTHER MULTILATERAL INSTITUTIONS': 'Other multilateral institutions',
    'FOUNDATIONS / PRIVATE DONORS': 'Foundations / private donors',
    'PRIVATE FUNDER': 'Private funder',
    'Other': 'Other funders (<3% of the year)',
}
REGION_LABELS = {'Other': 'Unspecified (developing countries)'}


def _yearly_by(df, col):
    """Summe pro Jahr und Gruppe in Million USD (Rohdaten in Thousand USD)"""
    d = df[df[col].notna()].copy()
    d['Amount'] = pd.to_numeric(d[FINANCE_COL], errors='coerce')
    d = d[d['Amount'] > 0]
    out = d.groupby(['Year', col])['Amount'].sum().reset_index()
    out['Year'] = out['Year'].astype(int)
    out['Amount'] = out['Amount'] / 1000
    return out


def comparison_table(cba_df, general_df, col):
    """Tabelle CBA vs. General Adaptation pro Jahr und Gruppe inkl. CBA-Anteil"""
    cba = _yearly_by(cba_df, col).rename(columns={'Amount': 'CBA_Million_USD'})
    gen = _yearly_by(general_df, col).rename(columns={'Amount': 'General_Adaptation_Million_USD'})
    t = gen.merge(cba, on=['Year', col], how='left').fillna({'CBA_Million_USD': 0})
    t['CBA_Share_Percent'] = 100 * t['CBA_Million_USD'] / t['General_Adaptation_Million_USD']
    return t.sort_values([col, 'Year'])


def _style_axes(fig, years, y_title):
    fig.update_xaxes(
        tickvals=list(range(min(years), max(years) + 1, 2)),
        tickfont=dict(family=FONT, size=12, color="#444"),
        showline=True, linecolor="#e5e5e5", showgrid=False
    )
    fig.update_yaxes(
        tickfont=dict(family=FONT, size=11, color="#444"),
        showline=True, linecolor="#e5e5e5",
        showgrid=True, gridcolor="#f0f0f0", rangemode='tozero'
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=FONT, color="#222"),
        hovermode='x unified'
    )
    if y_title:
        fig.update_yaxes(title_text=y_title, title_font=dict(family=FONT, size=14, color="#444"))


# --- Figure 1: Zeitreihe pro Kontinent ---------------------------------------

def _region_df(df):
    return df[(df['Recipient'] != 'REGIONAL') & (df['Region'].notna())]


def create_timeseries_region_comparison(cba_df, general_df):
    """Figure 1 im Originalstil, General Adaptation als gestrichelte Vergleichslinie je Kontinent"""
    cba = _yearly_by(_region_df(cba_df), 'Region')
    gen = _yearly_by(_region_df(general_df), 'Region')
    regions = list(cba['Region'].unique())

    fig = go.Figure()
    for i, region in enumerate(regions):
        color = CUSTOM_PAL[i % len(CUSTOM_PAL)]
        for data, dash, label in [(cba, 'solid', CBA_LABEL), (gen, 'dash', GENERAL_LABEL)]:
            r = data[data['Region'] == region]
            # Legende: ein Eintrag pro Kontinent (Farbe), Klick blendet beide Linien des Kontinents aus
            fig.add_trace(go.Scatter(
                x=r['Year'], y=r['Amount'], mode='lines+markers',
                name=REGION_LABELS.get(region, region), legendgroup=region, legendgrouptitle_text="Recipient continent" if i == 0 else None,
                line=dict(color=color, width=3 if dash == 'solid' else 2, dash=dash),
                marker=dict(size=7 if dash == 'solid' else 5, color=color),
                hovertemplate=f"{REGION_LABELS.get(region, region)} – {label}: %{{y:,.0f}} M USD<extra></extra>",
                showlegend=(dash == 'solid')
            ))
    # Legende: Linienart
    for label, dash in [(CBA_LABEL, 'solid'), (GENERAL_LABEL, 'dash')]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', name=label,
            legendgroup='style', legendgrouptitle_text="Line style",
            line=dict(color="#444", width=3 if dash == 'solid' else 2, dash=dash)
        ))
    _style_axes(fig, cba['Year'], "Finance commitments (Million USD, 2023)")
    fig.update_xaxes(title_text="Year", title_font=dict(family=FONT, size=14, color="#444"))
    fig.update_layout(
        margin=dict(l=60, r=30, t=40, b=60),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    groupclick="togglegroup", tracegroupgap=18,
                    font=dict(family=FONT, size=12, color="#222"))
    )
    return fig


def create_timeseries_region_small_multiples(cba_df, general_df):
    """Figure 1 als Small Multiples: ein Panel pro Kontinent, eigene y-Achse"""
    cba = _yearly_by(_region_df(cba_df), 'Region')
    gen = _yearly_by(_region_df(general_df), 'Region')
    regions = list(gen.groupby('Region')['Amount'].sum().sort_values(ascending=False).index)
    return _small_multiples(cba, gen, 'Region', regions, cols=4)


# --- Figure 2: Zeitreihe pro Funder Category ---------------------------------

def create_timeseries_donor_comparison_total(cba_df, general_df):
    """Figure 2 im Originalstil (gestapelt) plus Gesamtlinie der General Adaptation Commitments"""
    cba = _yearly_by(cba_df, 'Funder Category')
    gen_total = _yearly_by(general_df, 'Funder Category').groupby('Year')['Amount'].sum()

    result = []
    for year, group in cba.groupby('Year'):
        g = group_small_categories(group.copy(), 'Funder Category', 'Amount', threshold=0.03)
        g['Year'] = year
        result.append(g)
    pivot = pd.concat(result, ignore_index=True).pivot_table(
        index='Year', columns='Funder Category', values='Amount', fill_value=0
    ).sort_index()
    cats = list(pivot.columns)
    colors = {cat: CUSTOM_PAL[i % len(CUSTOM_PAL)] for i, cat in enumerate(cats)}
    if 'Other' in cats:
        colors['Other'] = "#cccccc"

    fig = go.Figure()
    for cat in cats:
        label = FUNDER_LABELS.get(cat, cat)
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[cat], mode='lines', stackgroup='one', name=label,
            legendgroup='cba', legendgrouptitle_text="CBA-framed commitments (stacked)",
            line=dict(width=0.5, color=colors[cat]),
            hovertemplate=f"CBA-framed – {label}: %{{y:,.0f}} M USD<extra></extra>"
        ))
    fig.add_trace(go.Scatter(
        x=gen_total.index, y=gen_total.values, mode='lines+markers', name="Total (all funders)",
        legendgroup='gen', legendgrouptitle_text="General adaptation commitments",
        line=dict(color="#222", width=2.5, dash='dash'), marker=dict(size=6, color="#222"),
        hovertemplate="General adaptation – total: %{y:,.0f} M USD<extra></extra>"
    ))
    _style_axes(fig, pivot.index, "Finance commitments (Million USD, 2023)")
    fig.update_xaxes(title_text="Year", title_font=dict(family=FONT, size=14, color="#444"))
    fig.update_layout(
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, tracegroupgap=18,
                    bgcolor='rgba(245,245,245,0.8)', bordercolor="#eee", borderwidth=1)
    )
    return fig


def create_timeseries_donor_small_multiples(cba_df, general_df):
    """Figure 2 als Small Multiples: ein Panel pro Funder Category, eigene y-Achse"""
    cba = _yearly_by(cba_df, 'Funder Category')
    gen = _yearly_by(general_df, 'Funder Category')
    cats = list(gen.groupby('Funder Category')['Amount'].sum().sort_values(ascending=False).index)
    return _small_multiples(cba, gen, 'Funder Category', cats, cols=3)


def _small_multiples(cba, gen, col, groups, cols):
    rows = -(-len(groups) // cols)
    labels = REGION_LABELS if col == 'Region' else FUNDER_LABELS
    titles = [labels.get(g, g) for g in groups]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.14)
    for i, grp in enumerate(groups):
        row, c = i // cols + 1, i % cols + 1
        g = gen[gen[col] == grp]
        k = cba[cba[col] == grp]
        fig.add_trace(go.Scatter(
            x=g['Year'], y=g['Amount'], mode='lines', name=GENERAL_LABEL,
            line=dict(color="#999", width=2, dash='dash'),
            fill='tozeroy', fillcolor='rgba(200,200,200,0.25)',
            showlegend=(i == 0), legendgroup='gen'
        ), row=row, col=c)
        fig.add_trace(go.Scatter(
            x=k['Year'], y=k['Amount'], mode='lines+markers', name=CBA_LABEL,
            line=dict(color=CUSTOM_PAL[0], width=3), marker=dict(size=5, color=CUSTOM_PAL[0]),
            showlegend=(i == 0), legendgroup='cba'
        ), row=row, col=c)
    _style_axes(fig, gen['Year'], None)
    fig.update_annotations(font=dict(family=FONT, size=13, color="#222"))
    fig.update_layout(
        height=330 * rows, margin=dict(l=60, r=30, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    fig.add_annotation(text="Million USD (2023)", textangle=-90, showarrow=False,
                       xref="paper", yref="paper", x=-0.045, y=0.5,
                       font=dict(family=FONT, size=14, color="#444"))
    return fig


# --- Figure 4: Weltkarte ------------------------------------------------------

def _map_layout(fig, colorbar_title):
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth',
                 bgcolor="white", landcolor="white", lakecolor="white"),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family=FONT, color="#222"),
        margin=dict(l=40, r=40, t=40, b=40),
        coloraxis_colorbar=dict(title=dict(text=colorbar_title), thickness=18, len=0.5,
                                outlinewidth=0, x=0.98, y=0.5),
        height=700
    )
    return fig


GREEN_GRADIENT = [
    [0.0, "#f8ffe8"], [0.15, "#bfd46f"], [0.5, "#81aa80"], [0.85, "#4b6b3c"], [1.0, "#1b2e13"]
]


def map_table(cba_df, general_df):
    gen = general_df.groupby('Recipient')[FINANCE_COL].sum() / 1000
    cba = cba_df.groupby('Recipient')[FINANCE_COL].sum() / 1000
    t = pd.DataFrame({'General_Adaptation_Million_USD': gen, 'CBA_Million_USD': cba}).fillna(0)
    t = t[t['General_Adaptation_Million_USD'] > 0]
    t['CBA_Share_Percent'] = 100 * t['CBA_Million_USD'] / t['General_Adaptation_Million_USD']
    return t.reset_index().rename(columns={'index': 'Recipient'}).sort_values(
        'General_Adaptation_Million_USD', ascending=False)


def create_map_general_finance(cba_df, general_df):
    """Figure 4 Vergleich: General Adaptation Commitments pro Empfängerland (gleicher Stil)"""
    t = map_table(cba_df, general_df)
    fig = px.choropleth(
        t, locations='Recipient', locationmode='country names',
        color='General_Adaptation_Million_USD', hover_name='Recipient',
        hover_data={'CBA_Million_USD': ':.1f', 'CBA_Share_Percent': ':.1f', 'Recipient': False},
        color_continuous_scale=GREEN_GRADIENT,
        labels={'General_Adaptation_Million_USD': 'General adaptation (Million USD)',
                'CBA_Million_USD': 'CBA (Million USD)', 'CBA_Share_Percent': 'CBA share (%)'}
    )
    return _map_layout(fig, "Million USD")


def create_map_cba_share(cba_df, general_df, min_general_million=50):
    """
    Figure 4 Vergleich: Anteil CBA-framed an General Adaptation Commitments (%) pro Land.
    Länder mit sehr kleinem Gesamtvolumen werden ausgeblendet, da der Anteil dort kaum aussagekräftig ist.
    """
    t = map_table(cba_df, general_df)
    t = t[t['General_Adaptation_Million_USD'] >= min_general_million]
    fig = px.choropleth(
        t, locations='Recipient', locationmode='country names',
        color='CBA_Share_Percent', hover_name='Recipient',
        hover_data={'CBA_Million_USD': ':.1f', 'General_Adaptation_Million_USD': ':.1f', 'Recipient': False},
        color_continuous_scale=GREEN_GRADIENT, range_color=(0, t['CBA_Share_Percent'].quantile(0.98)),
        labels={'CBA_Share_Percent': 'CBA share (%)', 'CBA_Million_USD': 'CBA (Million USD)',
                'General_Adaptation_Million_USD': 'General adaptation (Million USD)'}
    )
    return _map_layout(fig, "CBA share of<br>adaptation (%)")


def create_all_comparison_plots(cba_df, general_df, save_html=True):
    plots = {
        'fig1_timeseries_region_comparison': create_timeseries_region_comparison(cba_df, general_df),
        'fig1_timeseries_region_small_multiples': create_timeseries_region_small_multiples(cba_df, general_df),
        'fig2_donor_timeseries_comparison_total': create_timeseries_donor_comparison_total(cba_df, general_df),
        'fig2_donor_timeseries_small_multiples': create_timeseries_donor_small_multiples(cba_df, general_df),
        'fig4_map_general_adaptation': create_map_general_finance(cba_df, general_df),
        'fig4_map_cba_share': create_map_cba_share(cba_df, general_df),
    }
    if save_html:
        for name, fig in plots.items():
            filename = f'output/{name}.html'
            fig.write_html(filename)
            print(f"Gespeichert: {filename}")
        comparison_table(_region_df(cba_df), _region_df(general_df), 'Region').to_csv(
            'output/comparison_region_by_year.csv', index=False)
        comparison_table(cba_df, general_df, 'Funder Category').to_csv(
            'output/comparison_funder_by_year.csv', index=False)
        map_table(cba_df, general_df).to_csv('output/comparison_by_country.csv', index=False)
        print("Gespeichert: output/comparison_*.csv")
    return plots
