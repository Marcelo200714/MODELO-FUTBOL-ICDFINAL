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

# NOTA: aquí existía un diccionario FIFA_RANKING escrito a mano (un único
# snapshot desactualizado), usado tanto como feature de entrenamiento como
# para una corrección exponencial manual sobre la predicción. Ese diccionario
# estaba desincronizado con el ranking real del propio dataset (ej. tenía a
# Croacia en el puesto 6 y a Ghana en el 43, cuando el histórico real los
# ubica en el 11 y el 76 respectivamente), lo que producía tablas de grupo
# sin sentido (selecciones fuertes como Inglaterra terminando últimas). Se
# eliminó por completo: el ranking ahora se lee siempre del histórico real
# (ver get_features_partido) y ya no se aplica ninguna corrección manual.

# Función que construye el vector de 16 features (idéntica a la del entrenamiento)
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

    # Ranking real (último valor observado en el histórico para ese equipo),
    # no un diccionario inventado.
    rank_h = rh['home_rank'].tail(1).values[0] if len(rh) else 50
    rank_a = ra['away_rank'].tail(1).values[0] if len(ra) else 50
    rank_diff_val = rank_h - rank_a
    rank_diff_log_val = np.sign(rank_diff_val) * np.log1p(np.abs(rank_diff_val))

    tier_h = rh['home_rank_tier'].tail(1).values[0] if len(rh) else 2
    tier_a = ra['away_rank_tier'].tail(1).values[0] if len(ra) else 2
    tier_diff_val = tier_h - tier_a

    net_last5_diff = fh['net_last5'] - fa['net_last5']

    feats = [
        elo_diff, overall_rating_diff, pace_diff_val,
        fh['gf_last5'], fh['gc_last5'], fa['gf_last5'], fa['gc_last5'],
        home_last5_wr, away_last5_wr,
        h2h_wr, h2h_gd,
        rank_diff_val, tier_diff_val,
        rank_diff_log_val,
        net_last5_diff,
        neutral_val
    ]
    return np.array(feats).reshape(1, -1)

# --------------------------------------------
# 3. Función de predicción (inferencia simétrica, sin corrección manual)
# --------------------------------------------
def predict_match(home_es, away_es):
    """Predice el marcador promediando dos inferencias simétricas
    (intercambiando quién figura como local), sin ninguna corrección manual
    posterior. El marcador final se muestrea de una distribución de Poisson
    centrada en los goles esperados que predice el regresor (técnica estándar
    para simular partidos a partir de un modelo de goles esperados): redondear
    directamente el promedio produce demasiados empates artificiales cuando
    dos equipos tienen valores esperados muy parecidos (ej. 1.31 vs 1.35
    siempre redondeaba a un empate 1-1), aplanando diferencias reales de
    nivel en la tabla de grupos."""
    X_s1 = get_features_partido(home_es, away_es, df_final, forma, neutral_val=1)
    g1_s1 = model_home.predict(X_s1)[0]
    g2_s1 = model_away.predict(X_s1)[0]

    X_s2 = get_features_partido(away_es, home_es, df_final, forma, neutral_val=1)
    g2_s2 = model_home.predict(X_s2)[0]
    g1_s2 = model_away.predict(X_s2)[0]

    lam1 = max(0.05, (g1_s1 + g1_s2) / 2.0)
    lam2 = max(0.05, (g2_s1 + g2_s2) / 2.0)

    return int(np.random.poisson(lam1)), int(np.random.poisson(lam2))

# --------------------------------------------
# 4. Simular fase de grupos
# --------------------------------------------
predicted_results = []
for _, row in df_2026.iterrows():
    home = row['home_team']
    away = row['away_team']
    gh, ga = predict_match(home, away)
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
print("\n--- TABLAS DE GRUPOS PREDICHAS ---")
for group, teams in sorted(group_tables.items()):
    print(f"\nGRUPO {group}")
    for pos, (team, stats) in enumerate(teams, start=1):
        print(f"{pos}. {team} (Pts:{stats['Pts']}, DG:{stats['DG']}, GF:{stats['GF']})")