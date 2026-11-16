import pandas as pd
import numpy as np

# Continent Mapping
CONTINENT_MAPPING = {
    'Africa': [
        'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
        'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 
        'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea',
        'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau',
        'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi',
        'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
        'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles',
        'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Swaziland',
        'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe',
        # Regional
        'Africa, regional', 'Eastern Africa, regional', 'Middle Africa, regional',
        'North of Sahara, regional', 'South of Sahara, regional', 'Southern Africa, regional',
        'Western Africa, regional'
    ],
    'Asia': [
        'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan',
        'Brunei', 'Cambodia', 'China', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq',
        'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos',
        'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea',
        'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia',
        'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand',
        'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan',
        'Vietnam', 'Yemen',
        # Regional
        'Asia, regional', 'Central Asia, regional', 'Far East Asia, regional',
        'Middle East, regional', 'South & Central Asia, regional', 'South Asia, regional'
    ],
    'Europe': [
        'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
        'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia',
        'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland',
        'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg',
        'Macedonia', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands',
        'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia',
        'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine',
        'United Kingdom', 'Vatican City',
        # Regional
        'Europe, regional'
    ],
    'North America': [
        'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica',
        'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala',
        'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama',
        'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines',
        'Trinidad and Tobago', 'United States',
        # Regional
        'America, regional', 'Caribbean & Central America, regional', 'Caribbean, regional',
        'Central America, regional'
    ],
    'South America': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana',
        'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela',
        # Regional
        'South America, regional'
    ],
    'Oceania': [
        'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru',
        'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands',
        'Tonga', 'Tuvalu', 'Vanuatu',
        # Regional
        'Oceania, regional', 'Cook Islands', 'Niue', 'Tokelau'
    ]
}

# Special category für unspezifizierte Entwicklungsländer
CONTINENT_MAPPING['Other'] = [
    'Developing countries, unspecified'
]

COUNTRY_NAME_MAPPING = {
    "Bolivia": "Bolivia",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Bolivia, Plurinational State of": "Bolivia",
    
    # China Varianten
    "China (People's Republic of)": "China",
    
    # Türkiye/Turkey
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    
    # Vietnam
    "Viet Nam": "Vietnam",
    "Vietnam": "Vietnam",
    
    # Venezuela
    "Venezuela": "Venezuela",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Venezuela, Bolivarian Republic o": "Venezuela",
    
    # Iran
    "Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    "Iran, Islamic Republic of": "Iran",
    
    # Laos
    "Lao People's Democratic Republic": "Laos",
    "Lao PDR": "Laos",
    "Laos": "Laos",
    
    # Kongo
    "Congo": "Congo",
    "Congo, the Democratic Republic o": "Congo, Dem. Rep.",
    "Congo (Democratic Republic of the)": "Congo, Dem. Rep.",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    
    # Elfenbeinküste
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
    
    # Syrien
    "Syria": "Syrian Arab Republic",
    "Syrian Arab Republic": "Syrian Arab Republic",
    
    # Myanmar
    "Myanmar": "Myanmar",
    "Burma": "Myanmar",
    
    # Nordkorea
    "North Korea": "Korea, Dem. People's Rep.",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    "Korea, Democratic People's Repub": "Korea, Dem. People's Rep.",
    
    # Südkorea
    "South Korea": "Korea, Republic of",
    "Korea": "Korea, Republic of",
    "Korea, Republic of": "Korea, Republic of",
    
    # Tansania
    "Tanzania": "Tanzania, United Republic of",
    "Tanzania, United Republic of": "Tanzania, United Republic of",
    
    # Ägypten
    "Egypt": "Egypt",
    "Egypt, Arab Rep.": "Egypt",
    
    # Gambia
    "Gambia": "Gambia",
    "Gambia, The": "Gambia",
    "Gambia (The)": "Gambia",
    
    # Mikronesien
    "Micronesia": "Micronesia, Federated States of",
    "Micronesia (Federated States of)": "Micronesia, Federated States of",
    "Micronesia, Fed. Sts.": "Micronesia, Federated States of",
    
    # Mazedonien
    "Macedonia": "North Macedonia",
    "North Macedonia": "North Macedonia",
    
    # Moldova
    "Moldova": "Moldova, Republic of",
    "Moldova, Republic of": "Moldova, Republic of",
    
    # West Bank
    "West Bank and Gaza Strip": "West Bank and Gaza",
    "West Bank and Gaza": "West Bank and Gaza",
    
    # Cabo Verde
    "Cape Verde": "Cabo Verde",
    "Cabo Verde": "Cabo Verde",
    
    # Eswatini
    "Swaziland": "Eswatini",
    "Eswatini": "Eswatini",
    
    # Timor
    "East Timor": "Timor-Leste",
    "Timor-Leste": "Timor-Leste",
}

# Recipient wird vor get_continent per COUNTRY_NAME_MAPPING umbenannt,
# daher die umbenannten Namen ebenfalls im Continent Mapping führen
for _continent, _countries in CONTINENT_MAPPING.items():
    CONTINENT_MAPPING[_continent] = list(dict.fromkeys(
        _countries + [COUNTRY_NAME_MAPPING.get(c, c) for c in _countries]
    ))
CONTINENT_MAPPING['Africa'] += ['Saint Helena']
CONTINENT_MAPPING['Asia'] += ['West Bank and Gaza']
CONTINENT_MAPPING['Europe'] += ['States Ex-Yugoslavia unspecified']
CONTINENT_MAPPING['North America'] += ['Montserrat']
CONTINENT_MAPPING['Oceania'] += [
    'Wallis and Futuna', 'Melanesia, regional', 'Micronesia, regional', 'Polynesia, regional'
]


# Funder Category Mapping
FUNDER_CATEGORIES = {
    'BILATERAL FUNDERS': [
        'Australia', 'Austria', 'Belgium', 'Canada', 'Czech Republic', 'Czechia', 'Denmark',
        'Estonia', 'Finland', 'France',
        'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Japan', 'Korea', 'Latvia',
        'Liechtenstein', 'Lithuania', 'Luxembourg', 'Monaco', 'Netherlands', 'New Zealand',
        'Norway', 'Poland', 'Portugal', 'Romania', 'Slovenia', 'Slovak Republic', 'Spain', 'Sweden',
        'Switzerland', 'United Arab Emirates', 'United Kingdom', 'United States', 'Qatar'
    ],
    'MULTILATERAL DEVELOPMENT BANKS': [
        'African Development Bank', 'African Development Fund',
        'Asian Infrastructure Investment Bank', 'Asian Development Bank',
        'Caribbean Development Bank', 'Central American Bank for Economic Integration',
        'Council of Europe Development Bank', 'Development Bank of Latin America',
        'European Bank for Reconstruction and Development',
        'EU Institutions (EIB)',
        'Inter-American Development Bank',
        'IDB Invest',
        'International Bank for Reconstruction and Development',
        'International Development Association',
        'International Finance Corporation',
        'Islamic Development Bank',
        'Black Sea Trade & Development Bank'
    ],
    'CLIMATE FUNDS': [
        'Adaptation Fund',
        'GEF Least Developed Counties Trust Fund (LDCF)',
        'GEF Least Developed Countries Trust Fund (LDCF)',
        'GEF Special Climate Change Trust Fund (SCCF)',
        'Global Environment Facility General Trust Fund',
        'Green Climate Fund', 'GCF',
        'Climate Investment Funds - Clean Technology Fund',
        'Climate Investment Funds - Strategic Climate Fund',
    ],
    'OTHER MULTILATERAL INSTITUTIONS': [
        'EU Institutions (excl. EIB)',
        'Global Green Growth Institute',
        'International Fund for Agricultural Development',
        'Nordic Development Fund',
        'Food and Agriculture Organisation',
        'CGIAR',
    ],
    'FOUNDATIONS / PRIVATE DONORS': [
        'Bezos Earth Fund',
        'Bloomberg Family Foundation',
        'Carnegie Corporation of New York',
        "Children's Investment Fund Foundation",
        'Citi Foundation',
        'David and Lucile Packard Foundation',
        'Ford Foundation',
        'Gates Foundation',
        'German Postcode Lottery',
        'IKEA Foundation',
        'John D. and Catherine T. MacArthur Foundation',
        'Laudes Foundation',
        'MAVA Foundation',
        'Margaret A. Cargill Foundation',
        'Mastercard Foundation',
        'Oak Foundation',
        'Open Society Foundations',
        'Rockefeller Foundation',
        'Swedish Postcode Lottery',
        'Dutch Postcode Lottery',
        'UBS Optimus Foundation',
        'Wellcome Trust',
        'William and Flora Hewlett Foundation',
        'Gordon and Betty Moore Foundation',
        'H&M Foundation',
        'Leona M. and Harry B. Helmsley Charitable Trust',
        'McKnight Foundation',
        "People's Postcode Lottery",
        'Van Leer Foundation',
        'Fondation Botnar',
        'Conrad N. Hilton Foundation',
        'Arcadia Fund',
        'La Caixa Banking Foundation'
    ]
}

# Aktualisierte Keyword Dimensionen
KEYWORDS = {
    'dimension_a': [
        'community', 'communities', 'communal', 'local', 'village', 'district', 
        'municipal', 'smallholder', 'pastoral', 'subsistence', 'indigenous', 'tribal', 
        'bottom-up', 'grassroot', 'home', 'household', 'slum', 'town', 'SMEs', 
        'cooperative', 'province', 'rural', 'civic'
    ],
    'dimension_b': [
        'participat', 'inclusi', 'deliberat', 'collaborat'
    ],
    'dimension_c': [
        'decentral', 'devol', 'polycentric', 'co-govern'
    ],
    'dimension_d': [
        'collective', 'commons', 'commoning', 'co-manage', 'co-produc', 'co-creat', 'need-based'
    ]
}

def get_continent(country):
    """Ermittelt den Kontinent basierend auf dem Land"""
    if pd.isna(country):
        return None
    
    country = str(country).strip()
    
    for continent, countries in CONTINENT_MAPPING.items():
        if country in countries:
            return continent
    
    return 'Other'

def get_funder_category(provider):
    """Ermittelt die Funder Category basierend auf dem Provider Namen"""
    if pd.isna(provider):
        return None
    
    provider = str(provider).strip().lower()
    
    # case-insensitive, da OECD z.B. 'EU institutions (EIB)' statt 'EU Institutions (EIB)' schreibt
    for category, funders in FUNDER_CATEGORIES.items():
        if provider in [f.lower() for f in funders]:
            return category
    
    return 'PRIVATE FUNDER'

def check_keywords_in_text(text, keywords):
    """
    Prüft ob Keywords aus einer Liste im Text vorkommen.
    Erkennt Keywords als Präfix, Suffix und Infix (Teilstring).
    Gibt zurück: (boolean: mindestens ein Match, liste_der_gefundenen_keywords)
    """
    if pd.isna(text):
        return False, []
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            found_keywords.append(keyword)
    
    return len(found_keywords) > 0, found_keywords

def extract_matching_words_from_text(text, keywords):
    """
    Findet und extrahiert die tatsächlichen Wörter aus dem Text, die zu den Keywords passen.
    Gibt zurück: (boolean: mindestens ein Match, liste_der_originalen_wörter_aus_text)
    """
    if pd.isna(text):
        return False, []
    
    import re
    
    text_lower = text.lower()
    found_words = []
    
    # Extrahiere alle Wörter aus dem Text
    all_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Finde alle Wörter im Text, die das Keyword enthalten
        for word in all_words:
            if keyword_lower in word.lower() and word.lower() not in [w.lower() for w in found_words]:
                found_words.append(word)
    
    return len(found_words) > 0, found_words

def filter_by_keywords(df):
    """
    Filtert DataFrame basierend auf neuer Logik und speichert gefundene originale Wörter
    """
    # Prüfe jede Dimension und sammle originale Wörter
    df['has_a'], df['words_a'] = zip(*df['Combined_Text'].apply(
        lambda x: extract_matching_words_from_text(x, KEYWORDS['dimension_a'])
    ))
    df['has_b'], df['words_b'] = zip(*df['Combined_Text'].apply(
        lambda x: extract_matching_words_from_text(x, KEYWORDS['dimension_b'])
    ))
    df['has_c'], df['words_c'] = zip(*df['Combined_Text'].apply(
        lambda x: extract_matching_words_from_text(x, KEYWORDS['dimension_c'])
    ))
    df['has_d'], df['words_d'] = zip(*df['Combined_Text'].apply(
        lambda x: extract_matching_words_from_text(x, KEYWORDS['dimension_d'])
    ))
    
    # Kombiniere alle gefundenen originalen Wörter
    df['Found_Original_Words'] = df.apply(
        lambda row: row['words_a'] + row['words_b'] + row['words_c'] + row['words_d'], 
        axis=1
    )
    
    # Zähle in wie vielen verschiedenen Dimensionen B, C, D Keywords gefunden wurden
    df['dimensions_bcd_count'] = df['has_b'].astype(int) + df['has_c'].astype(int) + df['has_d'].astype(int)
    
    # Filter: Dimension A ODER mindestens 2 verschiedene Dimensionen aus B, C, D
    mask = (df['has_a']) | (df['dimensions_bcd_count'] >= 2)
    
    df_filtered = df[mask].copy()
    
    # Behalte die Flags als binäre Spalten (0/1)
    df_filtered['Has_Dimension_A'] = df_filtered['has_a'].astype(int)
    df_filtered['Has_Dimension_B'] = df_filtered['has_b'].astype(int)
    df_filtered['Has_Dimension_C'] = df_filtered['has_c'].astype(int)
    df_filtered['Has_Dimension_D'] = df_filtered['has_d'].astype(int)
    
    # Entferne nur die temporären Listen-Spalten, aber nicht die Flags
    df_filtered = df_filtered.drop(columns=[
        'has_a', 'words_a', 'has_b', 'words_b', 'has_c', 'words_c',
        'has_d', 'words_d', 'dimensions_bcd_count'
    ])
    
    return df_filtered

def load_vulnerability(filepath):
    """Lädt Vulnerability Datensatz"""
    df = pd.read_csv(filepath, sep=',', encoding='latin-1')
    # Country Name Mapping anwenden
    df['Name'] = df['Name'].replace(COUNTRY_NAME_MAPPING)
    return df

def load_population(filepath):
    """Lädt Population Datensatz"""
    df = pd.read_csv(filepath, sep=',', encoding='latin-1')
    # Country Name Mapping anwenden
    df['Country Name'] = df['Country Name'].replace(COUNTRY_NAME_MAPPING)
    return df

def preprocess_crdf(filepath, apply_keyword_filter=True):
    """Lädt und verarbeitet CRDF-RP Datensatz

    apply_keyword_filter=False liefert die General Adaptation Commitments
    (Principal-Marker + GCF manuell auf Principal, ohne CBA Keyword Filter).
    """
    df = pd.read_csv(filepath, sep=';', encoding='latin-1', quotechar='"', low_memory=False)
    
    df = df[df['Year'] >= 2011].copy()

    # Country Name Mapping anwenden
    df['Recipient'] = df['Recipient'].replace(COUNTRY_NAME_MAPPING)
    
    # Bereinige Finance-Spalte: Ersetze Komma durch Punkt (Dezimaltrennzeichen)
    finance_col = 'Adaptation-related development finance (includes overlap) - Commitment - 2023 USD thousand'
    df[finance_col] = df[finance_col].astype(str).str.replace(',', '.').str.strip()
    
    # Konvertiere Finanzierungsspalte zu numerisch
    df[finance_col] = pd.to_numeric(df[finance_col], errors='coerce')
    
    # Principally adaptation: 'Adaptation objective' (nicht 'Climate objective', das auch
    # Mitigation-Principal umfasst) + MDB Climate components + alle GCF Commitments
    # (manuell als principally adaptation gesetzt, Savvidou et al. 2021)
    adaptation_col = 'Adaptation objective (applies to Rio-marked data only)'
    climate_col = 'Climate objective (applies to Rio-marked data only) or climate component'
    is_gcf = df['Provider'] == 'GCF'
    df.loc[is_gcf, adaptation_col] = 'Principal'
    df = df[(df[adaptation_col] == 'Principal') | (df[climate_col] == 'Climate components')].copy()
    
    df['Combined_Text'] = df['Project Title'].fillna('') + ' ' + df['Description'].fillna('')
    df['Combined_Text'] = df['Combined_Text'].str.strip()
    
    df['Funder Category'] = df['Provider (detailed)'].apply(get_funder_category)
    
    # Region Spalte hinzufügen
    df['Region'] = df['Recipient'].apply(get_continent)
    
    if apply_keyword_filter:
        df = filter_by_keywords(df)
    
    return df

def load_all_datasets(vulnerability_path, population_path, crdf_path):
    """Lädt alle drei Datensätze"""
    vulnerability_df = load_vulnerability(vulnerability_path)
    population_df = load_population(population_path)
    crdf_df = preprocess_crdf(crdf_path)
    
    return vulnerability_df, population_df, crdf_df