import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Rutas (relativas a la carpeta raíz del proyecto)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --------------------------------------------
# 1. Cargar modelos y datos
# --------------------------------------------
with open(os.path.join(MODELS_DIR, 'model_home.pkl'), 'rb') as f:
    model_home = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'model_away.pkl'), 'rb') as f:
    model_away = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'df_final.pkl'), 'rb') as f:
    df_final = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'forma.pkl'), 'rb') as f:
    forma = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'player_stats.pkl'), 'rb') as f:
    player_stats = pickle.load(f)

# Cargar datos de mundiales para obtener los grupos 2026
df_wc = pd.read_csv(os.path.join(DATA_DIR, 'worldcup_matches.csv'), parse_dates=['date'])
df_2026 = df_wc[(df_wc['year'] == 2026) & (df_wc['stage'] == 'Group Stage')].copy()
df_2026 = df_2026.sort_values(['group', 'date'])

# --------------------------------------------
# 2. Diccionario y funciones auxiliares (idénticas al entrenamiento)
# --------------------------------------------
ES_TO_EN = {
    'Argentina': 'Argentina', 'Brasil': 'Brazil', 'México': 'Mexico',
    'Uruguay': 'Uruguay', 'Colombia': 'Colombia', 'Chile': 'Chile',
    'Ecuador': 'Ecuador', 'Perú': 'Peru', 'Venezuela': 'Venezuela',
    'Paraguay': 'Paraguay', 'Bolivia': 'Bolivia', 'Costa Rica': 'Costa Rica',
    'Panamá': 'Panama', 'Honduras': 'Honduras', 'El Salvador': 'El Salvador',
    'Guatemala': 'Guatemala', 'Jamaica': 'Jamaica',
    'Trinidad y Tobago': 'Trinidad and Tobago',
    'Haití': 'Haiti', 'Cuba': 'Cuba', 'Curaçao': 'Curacao',
    'Estados Unidos': 'United States', 'Canadá': 'Canada',
    'España': 'Spain', 'Francia': 'France', 'Alemania': 'Germany',
    'Italia': 'Italy', 'Portugal': 'Portugal', 'Inglaterra': 'England',
    'Países Bajos': 'Netherlands', 'Bélgica': 'Belgium', 'Croacia': 'Croatia',
    'Dinamarca': 'Denmark', 'Suecia': 'Sweden', 'Noruega': 'Norway',
    'Polonia': 'Poland', 'Austria': 'Austria', 'Suiza': 'Switzerland',
    'Turquía': 'Turkey', 'Escocia': 'Scotland', 'Gales': 'Wales',
    'Hungría': 'Hungary', 'República Checa': 'Czech Republic',
    'Eslovaquia': 'Slovakia', 'Eslovenia': 'Slovenia', 'Serbia': 'Serbia',
    'Grecia': 'Greece', 'Rumania': 'Romania', 'Ucrania': 'Ukraine',
    'Rusia': 'Russia', 'Albania': 'Albania', 'Kosovo': 'Kosovo',
    'Irlanda': 'Republic of Ireland', 'Finlandia': 'Finland',
    'Islandia': 'Iceland', 'Bosnia': 'Bosnia and Herzegovina',
    'Macedonia del Norte': 'North Macedonia', 'Montenegro': 'Montenegro',
    'Azerbaiyán': 'Azerbaijan', 'Georgia': 'Georgia', 'Armenia': 'Armenia',
    'Marruecos': 'Morocco', 'Senegal': 'Senegal', 'Nigeria': 'Nigeria',
    'Ghana': 'Ghana', 'Camerún': 'Cameroon', 'Costa de Marfil': 'Ivory Coast',
    'Egipto': 'Egypt', 'Argelia': 'Algeria', 'Túnez': 'Tunisia',
    'Mali': 'Mali', 'Burkina Faso': 'Burkina Faso', 'Sudáfrica': 'South Africa',
    'Angola': 'Angola', 'Zambia': 'Zambia', 'Tanzania': 'Tanzania',
    'Etiopía': 'Ethiopia', 'Uganda': 'Uganda', 'Guinea': 'Guinea',
    'Mozambique': 'Mozambique', 'Zimbabue': 'Zimbabwe', 'Libia': 'Libya',
    'Gabón': 'Gabon', 'Congo': 'DR Congo', 'Benín': 'Benin',
    'Cabo Verde': 'Cape Verde',
    'Japón': 'Japan', 'Corea del Sur': 'South Korea',
    'Arabia Saudita': 'Saudi Arabia', 'Irán': 'IR Iran',
    'Australia': 'Australia', 'China': 'China', 'Catar': 'Qatar',
    'Emiratos Árabes Unidos': 'United Arab Emirates',
    'Uzbekistán': 'Uzbekistan', 'Irak': 'Iraq', 'Kuwait': 'Kuwait',
    'Baréin': 'Bahrain', 'Omán': 'Oman', 'Siria': 'Syria',
    'India': 'India', 'Vietnam': 'Vietnam', 'Tailandia': 'Thailand',
    'Indonesia': 'Indonesia', 'Filipinas': 'Philippines',
    'Nueva Zelanda': 'New Zealand', 'Jordania': 'Jordan',
    'Palestina': 'Palestine'
}
def traducir(nombre_es):
    return ES_TO_EN.get(nombre_es, nombre_es)

# Ranking FIFA (mismo diccionario que en entrenamiento)
FIFA_RANKING = {
    'Argentina': 1, 'France': 2, 'Brazil': 3, 'England': 4, 'Belgium': 5,
    'Croatia': 6, 'Netherlands': 7, 'Portugal': 8, 'Italy': 9, 'Spain': 10,
    'USA': 11, 'Mexico': 12, 'Germany': 13, 'Uruguay': 14, 'Switzerland': 15,
    'Colombia': 16, 'Senegal': 17, 'Denmark': 18, 'Japan': 19, 'Iran': 20,
    'South Korea': 21, 'Australia': 22, 'Ukraine': 23, 'Austria': 24, 'Sweden': 25,
    'Nigeria': 26, 'Ecuador': 27, 'Wales': 28, 'Peru': 29, 'Poland': 30,
    'Morocco': 31, 'Egypt': 32, 'Chile': 33, 'Tunisia': 34, 'Turkey': 35,
    'Costa Rica': 36, 'Russia': 37, 'Algeria': 38, 'Scotland': 39, 'Norway': 40,
    'Romania': 41, 'Czech Republic': 42, 'Ghana': 43, 'Qatar': 44, 'Saudi Arabia': 45,
    'Panama': 46, 'Honduras': 47, 'Jamaica': 48, 'Paraguay': 49, 'Ivory Coast': 50,
    'Cameroon': 51, 'South Africa': 52, 'Bolivia': 53, 'Venezuela': 54, 'Canada': 55,
    'New Zealand': 56, 'Uzbekistan': 57, 'Cape Verde': 58, 'DR Congo': 59, 'Haiti': 60,
    'Bosnia and Herzegovina': 61, 'Slovakia': 62, 'Slovenia': 63, 'Hungary': 64,
    'Greece': 65, 'Ireland': 66, 'Finland': 67, 'Iceland': 68, 'Albania': 69,
    'Montenegro': 70, 'North Macedonia': 71, 'Georgia': 72, 'Armenia': 73,
    'Kosovo': 74, 'Azerbaijan': 75, 'Jordan': 76, 'Iraq': 77, 'Oman': 78,
    'Bahrain': 79, 'Kuwait': 80, 'UAE': 81, 'Syria': 82, 'Palestine': 83,
    'India': 84, 'China': 85, 'Indonesia': 86, 'Thailand': 87, 'Vietnam': 88,
    'Philippines': 89, 'Cuba': 90, 'Curacao': 91, 'Trinidad and Tobago': 92,
    'El Salvador': 93, 'Guatemala': 94, 'Benin': 95, 'Burkina Faso': 96,
    'Mali': 97, 'Zambia': 98, 'Angola': 99, 'Uganda': 100, 'Ethiopia': 101,
    'Tanzania': 102, 'Zimbabwe': 103, 'Mozambique': 104, 'Libya': 105, 'Gabon': 106,
    'Guinea': 107, 'Guyana': 108, 'Barbados': 109, 'Mauritania': 110
}
def get_rank(team):
    return FIFA_RANKING.get(team, 100)

# Función que construye el vector de 17 features (idéntica a la del entrenamiento)
def get_features_partido(home_es, away_es, df_final, forma, neutral_val=1):
    home_en = traducir(home_es)
    away_en = traducir(away_es)

    row_home = df_final[df_final['home_team'] == home_en].tail(1)
    row_away = df_final[df_final['away_team'] == away_en].tail(1)
    elo_home = row_home['elo_home_pre'].values[0] if len(row_home) else 1500.0
    elo_away = row_away['elo_away_pre'].values[0] if len(row_away) else 1500.0
    elo_diff = elo_home - elo_away

    ph = player_stats[player_stats['team'] == home_en]
    pa = player_stats[player_stats['team'] == away_en]
    rating_home = ph['overall_rating_avg'].values[0] if len(ph) else 70.0
    rating_away = pa['overall_rating_avg'].values[0] if len(pa) else 70.0
    pace_home   = ph['pace_avg'].values[0]            if len(ph) else 70.0
    pace_away   = pa['pace_avg'].values[0]            if len(pa) else 70.0

    overall_rating_diff = rating_home - rating_away
    pace_diff_val       = pace_home   - pace_away

    fh = forma.get(home_en, {'gf_last5': 1.2, 'gc_last5': 1.0, 'net_last5': 0.0})
    fa = forma.get(away_en, {'gf_last5': 1.2, 'gc_last5': 1.0, 'net_last5': 0.0})

    rh = df_final[df_final['home_team'] == home_en]
    ra = df_final[df_final['away_team'] == away_en]
    home_last5_wr = rh['home_last5_winrate'].tail(1).values[0] if len(rh) else 0.5
    away_last5_wr = ra['away_last5_winrate'].tail(1).values[0] if len(ra) else 0.5

    h2h = df_final[(df_final['home_team'] == home_en) & (df_final['away_team'] == away_en)].tail(5)
    h2h_wr = h2h['h2h_last5_home_winrate'].mean() if len(h2h) else 0.5
    h2h_gd = h2h['h2h_last5_avg_gd'].mean()       if len(h2h) else 0.0

    rank_h = rh['home_rank'].tail(1).values[0] if len(rh) else 50
    rank_a = ra['away_rank'].tail(1).values[0] if len(ra) else 50
    rank_diff_val = rank_h - rank_a

    tier_h = rh['home_rank_tier'].tail(1).values[0] if len(rh) else 2
    tier_a = ra['away_rank_tier'].tail(1).values[0] if len(ra) else 2
    tier_diff_val = tier_h - tier_a

    # --- Nuevas features (como en entrenamiento) ---
    fifa_rank_h = get_rank(home_en)
    fifa_rank_a = get_rank(away_en)
    fifa_rank_diff = fifa_rank_h - fifa_rank_a
    fifa_rank_diff_log = np.sign(fifa_rank_diff) * np.log1p(np.abs(fifa_rank_diff))
    net_last5_diff = fh['net_last5'] - fa['net_last5']

    feats = [
        elo_diff, overall_rating_diff, pace_diff_val,
        fh['gf_last5'], fh['gc_last5'], fa['gf_last5'], fa['gc_last5'],
        home_last5_wr, away_last5_wr,
        h2h_wr, h2h_gd,
        rank_diff_val, tier_diff_val,
        fifa_rank_diff,
        fifa_rank_diff_log,
        net_last5_diff,
        neutral_val
    ]
    return np.array(feats).reshape(1, -1)

# --------------------------------------------
# 3. Función de predicción con corrección exponencial
# --------------------------------------------
def predict_match_corregido(home_es, away_es):
    # Obtener los goles base del modelo (sin corrección)
    X = get_features_partido(home_es, away_es, df_final, forma, neutral_val=1)
    g1_base = model_home.predict(X)[0]
    g2_base = model_away.predict(X)[0]

    # Ranking para corrección exponencial
    rank1 = get_rank(traducir(home_es))
    rank2 = get_rank(traducir(away_es))
    rank_diff = (rank2 - rank1) / 50.0   # positivo si home_es es mejor
    correction = np.exp(rank_diff * 0.8) # factor 0.8 (ajustable)

    g1 = max(0, int(round(g1_base * correction)))
    g2 = max(0, int(round(g2_base / correction if correction > 0 else g2_base)))

    # Si empate y gran diferencia, forzar un gol más al mejor
    if abs(rank_diff) > 0.3 and g1 == g2:
        if rank_diff > 0:
            g1 += 1
        else:
            g2 += 1

    return min(g1, 7), min(g2, 7)

# --------------------------------------------
# 4. Simular fase de grupos
# --------------------------------------------
predicted_results = []
for _, row in df_2026.iterrows():
    home = row['home_team']
    away = row['away_team']
    gh, ga = predict_match_corregido(home, away)
    predicted_results.append({
        'home_team': home,
        'away_team': away,
        'goals_home': gh,
        'goals_away': ga,
        'group': row['group']
    })

pred_df = pd.DataFrame(predicted_results)

# --------------------------------------------
# 5. Calcular tablas
# --------------------------------------------
groups = pred_df['group'].unique()
group_tables = {}

for group in sorted(groups):
    df_group = pred_df[pred_df['group'] == group]
    teams = set(df_group['home_team']).union(set(df_group['away_team']))
    records = {}
    for team in teams:
        records[team] = {'Pts': 0, 'GF': 0, 'GC': 0, 'DG': 0, 'PJ': 0}
    
    for _, match in df_group.iterrows():
        home, away = match['home_team'], match['away_team']
        gh, ga = match['goals_home'], match['goals_away']
        records[home]['GF'] += gh
        records[home]['GC'] += ga
        records[away]['GF'] += ga
        records[away]['GC'] += gh
        records[home]['PJ'] += 1
        records[away]['PJ'] += 1
        if gh > ga:
            records[home]['Pts'] += 3
        elif gh < ga:
            records[away]['Pts'] += 3
        else:
            records[home]['Pts'] += 1
            records[away]['Pts'] += 1
    
    for team in records:
        records[team]['DG'] = records[team]['GF'] - records[team]['GC']
    
    sorted_teams = sorted(records.items(), key=lambda x: (x[1]['Pts'], x[1]['DG'], x[1]['GF']), reverse=True)
    group_tables[group] = sorted_teams

# --------------------------------------------
# 6. Visualizar 4x3 con puntajes
# --------------------------------------------
def show_group_predictions(group_tables):
    groups = sorted(group_tables.keys())
    n_groups = len(groups)
    n_cols = 4
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i in range(n_groups, len(axes)):
        axes[i].axis('off')
    
    for idx, group in enumerate(groups):
        ax = axes[idx]
        ax.axis('off')
        ax.text(0.5, 0.92, f'GRUPO {group}', transform=ax.transAxes,
                fontsize=16, fontweight='bold', ha='center', va='center')
        
        teams = group_tables[group]
        y_start = 0.78
        step = 0.14
        for i, (team, stats) in enumerate(teams):
            if i < 2:
                color = 'green'
                weight = 'bold'
            elif i == len(teams)-1:
                color = 'red'
                weight = 'normal'
            else:
                color = 'black'
                weight = 'normal'
            label = f"{team} ({stats['Pts']} pts)"
            ax.text(0.5, y_start - i*step, label, transform=ax.transAxes,
                    fontsize=11, ha='center', va='center', color=color, fontweight=weight)
    
    plt.tight_layout()
    plt.show()

show_group_predictions(group_tables)

# --------------------------------------------
# 7. Mostrar en consola
# --------------------------------------------
print("\n--- TABLAS DE GRUPOS PREDICHAS (CORREGIDO) ---")
for group, teams in sorted(group_tables.items()):
    print(f"\nGRUPO {group}")
    for pos, (team, stats) in enumerate(teams, start=1):
        print(f"{pos}. {team} (Pts:{stats['Pts']}, DG:{stats['DG']}, GF:{stats['GF']})")