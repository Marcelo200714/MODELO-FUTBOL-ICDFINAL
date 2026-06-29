import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import os
import math
import itertools
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
    'Estados Unidos': 'USA', 'Canadá': 'Canada',
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
    'Ghana': 'Ghana', 'Camerún': 'Cameroon', 'Costa de Marfil': "Côte d'Ivoire",
    'Egipto': 'Egypt', 'Argelia': 'Algeria', 'Túnez': 'Tunisia',
    'Mali': 'Mali', 'Burkina Faso': 'Burkina Faso', 'Sudáfrica': 'South Africa',
    'Angola': 'Angola', 'Zambia': 'Zambia', 'Tanzania': 'Tanzania',
    'Etiopía': 'Ethiopia', 'Uganda': 'Uganda', 'Guinea': 'Guinea',
    'Mozambique': 'Mozambique', 'Zimbabue': 'Zimbabwe', 'Libia': 'Libya',
    'Gabón': 'Gabon', 'RD Congo': 'Congo DR', 'Benín': 'Benin',
    'Cabo Verde': 'Cabo Verde',
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
# 3. Función de predicción (inferencia simétrica, sin corrección manual,
#    sin aleatoriedad)
# --------------------------------------------
def _poisson_pmf(lam, max_goals):
    return [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(max_goals + 1)]

def match_probs(lam1, lam2, max_goals=10):
    """Probabilidad exacta de victoria/empate/derrota a partir de dos goles
    esperados (Poisson), calculada de forma analítica sobre todas las
    combinaciones de marcador posibles (no se muestrea nada al azar), por lo
    que el resultado es siempre el mismo para el mismo par de equipos."""
    p1 = _poisson_pmf(lam1, max_goals)
    p2 = _poisson_pmf(lam2, max_goals)
    p_home_win = sum(p1[i] * p2[j] for i in range(max_goals + 1) for j in range(i))
    p_draw     = sum(p1[i] * p2[i] for i in range(max_goals + 1))
    p_away_win = sum(p1[i] * p2[j] for i in range(max_goals + 1) for j in range(i + 1, max_goals + 1))
    return p_home_win, p_draw, p_away_win

def predict_match_expected(home_es, away_es):
    """Decide el resultado del partido (victoria/empate/derrota) de forma
    determinista: promedia dos inferencias simétricas (intercambiando quién
    figura como local) para obtener los goles esperados, deriva la
    probabilidad exacta de victoria/empate/derrota a partir de esos goles
    esperados (Poisson) y se queda con el desenlace más probable de los tres.
    Los puntos son los de siempre en fútbol (3/1/0) — no fracciones — y el
    marcador mostrado (goles redondeados) es coherente con ese desenlace.
    No hay ningún número aleatorio de por medio, así que el resultado es
    siempre el mismo para el mismo par de equipos (antes, simular un
    marcador con Poisson en cada corrida hacía que el mismo equipo pudiera
    terminar con 9 puntos en una corrida y 1 en la siguiente, solo por la
    varianza de una muestra aleatoria sobre 3 partidos)."""
    X_s1 = get_features_partido(home_es, away_es, df_final, forma, neutral_val=1)
    g1_s1 = model_home.predict(X_s1)[0]
    g2_s1 = model_away.predict(X_s1)[0]

    X_s2 = get_features_partido(away_es, home_es, df_final, forma, neutral_val=1)
    g2_s2 = model_home.predict(X_s2)[0]
    g1_s2 = model_away.predict(X_s2)[0]

    lam1 = max(0.05, (g1_s1 + g1_s2) / 2.0)
    lam2 = max(0.05, (g2_s1 + g2_s2) / 2.0)

    p_home_win, p_draw, p_away_win = match_probs(lam1, lam2)

    if p_draw >= p_home_win and p_draw >= p_away_win:
        gh = ga = round((lam1 + lam2) / 2.0)
        pts_home, pts_away = 1, 1
    elif p_home_win > p_away_win:
        gh, ga = round(lam1), round(lam2)
        if gh <= ga:
            gh = ga + 1
        pts_home, pts_away = 3, 0
    else:
        gh, ga = round(lam1), round(lam2)
        if ga <= gh:
            ga = gh + 1
        pts_home, pts_away = 0, 3

    return gh, ga, pts_home, pts_away

# --------------------------------------------
# 4. Simular fase de grupos
# --------------------------------------------
# worldcup_matches.csv no siempre trae los 6 partidos del round-robin de un
# grupo de 4 equipos (a algunos grupos les faltan 2 de los 6 cruces posibles),
# lo que dejaba a ciertos equipos con menos partidos jugados que otros y un
# tope de puntos distinto según el grupo. Se reconstruye el calendario
# completo (todas las combinaciones de 2 equipos por grupo) a partir de los
# equipos que sí aparecen listados, para que todos jueguen sus 3 partidos.
predicted_results = []
for group in sorted(df_2026['group'].unique()):
    dg = df_2026[df_2026['group'] == group]
    teams = sorted(set(dg['home_team']).union(set(dg['away_team'])))
    for home, away in itertools.combinations(teams, 2):
        gh, ga, pts_home, pts_away = predict_match_expected(home, away)
        predicted_results.append({
            'home_team': home,
            'away_team': away,
            'goals_home': gh,
            'goals_away': ga,
            'pts_home': pts_home,
            'pts_away': pts_away,
            'group': group
        })

pred_df = pd.DataFrame(predicted_results)

# --------------------------------------------
# 5. Calcular tablas (con valores esperados, no marcadores al azar)
# --------------------------------------------
groups = pred_df['group'].unique()
group_tables = {}

for group in sorted(groups):
    df_group = pred_df[pred_df['group'] == group]
    teams = set(df_group['home_team']).union(set(df_group['away_team']))
    records = {}
    for team in teams:
        records[team] = {'Pts': 0, 'GF': 0, 'GC': 0, 'DG': 0}

    for _, match in df_group.iterrows():
        home, away = match['home_team'], match['away_team']
        gh, ga = match['goals_home'], match['goals_away']
        records[home]['GF'] += gh
        records[home]['GC'] += ga
        records[away]['GF'] += ga
        records[away]['GC'] += gh
        records[home]['Pts'] += match['pts_home']
        records[away]['Pts'] += match['pts_away']

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
        print(f"{pos}. {team} (Pts:{stats['Pts']}, DG:{stats['DG']:+d}, GF:{stats['GF']})")