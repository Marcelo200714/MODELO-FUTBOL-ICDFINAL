# ============================================================================
# CELDA 1 - Librerías y carga
# ============================================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import random

# Configura rutas (cambia si es necesario)
PATH_MATCHES = 'national_matches_(1992-2026).csv'
PATH_PLAYERS = 'player-data-full.csv'
PATH_RESULTS = 'results.csv'
PATH_WC      = 'worldcup_matches.csv'

# Carga de datos
df_matches = pd.read_csv(PATH_MATCHES, encoding='utf-8')
df_players = pd.read_csv(PATH_PLAYERS, encoding='latin1')
df_results = pd.read_csv(PATH_RESULTS, encoding='latin1')
df_wc      = pd.read_csv(PATH_WC,      encoding='latin1')

df_matches['date'] = pd.to_datetime(df_matches['date'], format='mixed', dayfirst=True)
df_results['date'] = pd.to_datetime(df_results['date'], format='mixed', dayfirst=True)

print(f'Partidos nacionales  : {df_matches.shape}')
print(f'Datos de jugadores   : {df_players.shape}')
print(f'Resultados históricos: {df_results.shape}')
print(f'Partidos de Mundiales: {df_wc.shape}')

# ============================================================================
# CELDA 2 - Agrupación de jugadores (Top 15 por rating)
# ============================================================================
df_top15 = (
    df_players
    .sort_values('overall_rating', ascending=False)
    .groupby('country_name')
    .head(15)
)

player_stats = (
    df_top15
    .groupby('country_name')
    .agg(
        overall_rating_avg = ('overall_rating', 'mean'),
        pace_avg           = ('sprint_speed',   'mean')
    )
    .reset_index()
    .rename(columns={'country_name': 'team'})
)

print(player_stats.head(10))
print(f'Selecciones con datos de jugadores: {len(player_stats)}')

# ============================================================================
# CELDA 3 - Traducción y merge
# ============================================================================
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
    'Palestina': 'Palestine',
}

def traducir(nombre_es):
    return ES_TO_EN.get(nombre_es, nombre_es)

def merge_player_stats(df, player_stats):
    df = df.merge(
        player_stats.rename(columns={'team': 'home_team', 'overall_rating_avg': 'home_overall_rating', 'pace_avg': 'home_pace'}),
        on='home_team', how='left'
    )
    df = df.merge(
        player_stats.rename(columns={'team': 'away_team', 'overall_rating_avg': 'away_overall_rating', 'pace_avg': 'away_pace'}),
        on='away_team', how='left'
    )
    return df

df_final = merge_player_stats(df_matches, player_stats)
print(f'df_final shape: {df_final.shape}')
print(df_final[['home_team','away_team','home_overall_rating','home_pace']].head())

# ============================================================================
# CELDA 4 - Ranking FIFA y feature engineering (MEJORADO)
# ============================================================================
# --- RANKING FIFA ---
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

# --- Feature Engineering ---
df_final['overall_rating_diff'] = (
    df_final['home_overall_rating'].fillna(70) - df_final['away_overall_rating'].fillna(70)
)
df_final['pace_diff'] = (
    df_final['home_pace'].fillna(70) - df_final['away_pace'].fillna(70)
)

# Diferencia de ranking FIFA (lineal)
df_final['home_fifa_rank'] = df_final['home_team'].map(get_rank)
df_final['away_fifa_rank'] = df_final['away_team'].map(get_rank)
df_final['fifa_rank_diff'] = df_final['home_fifa_rank'] - df_final['away_fifa_rank']

# --- NUEVA: transformación logarítmica de la diferencia de ranking ---
# Captura mejor la no linealidad: equipos muy superiores tienen ventaja extra
df_final['fifa_rank_diff_log'] = np.sign(df_final['fifa_rank_diff']) * np.log1p(np.abs(df_final['fifa_rank_diff']))

# Forma reciente (últimos 5 partidos)
def calcular_forma(df_results, ventana=5):
    df_r = df_results.sort_values('date').copy()
    forma_dict = {}
    todos_equipos = pd.concat([df_r['home_team'], df_r['away_team']]).unique()
    for equipo in todos_equipos:
        como_local = df_r[df_r['home_team'] == equipo][['date','home_score','away_score']].rename(
            columns={'home_score':'gf','away_score':'gc'})
        como_visitante = df_r[df_r['away_team'] == equipo][['date','away_score','home_score']].rename(
            columns={'away_score':'gf','home_score':'gc'})
        partidos = pd.concat([como_local, como_visitante]).sort_values('date').tail(ventana)
        if len(partidos) > 0:
            forma_dict[equipo] = {
                'gf_last5': partidos['gf'].mean(),
                'gc_last5': partidos['gc'].mean(),
                'net_last5': (partidos['gf'] - partidos['gc']).mean()
            }
        else:
            forma_dict[equipo] = {'gf_last5': 1.0, 'gc_last5': 1.0, 'net_last5': 0.0}
    return forma_dict

forma = calcular_forma(df_results)
print(f'Equipos con forma calculada: {len(forma)}')

df_final['home_gf_last5'] = df_final['home_team'].map(lambda t: forma.get(t, {}).get('gf_last5', 1.0))
df_final['home_gc_last5'] = df_final['home_team'].map(lambda t: forma.get(t, {}).get('gc_last5', 1.0))
df_final['away_gf_last5'] = df_final['away_team'].map(lambda t: forma.get(t, {}).get('gf_last5', 1.0))
df_final['away_gc_last5'] = df_final['away_team'].map(lambda t: forma.get(t, {}).get('gc_last5', 1.0))

# --- NUEVA: diferencia de neto en últimos 5 partidos ---
df_final['home_net_last5'] = df_final['home_team'].map(lambda t: forma.get(t, {}).get('net_last5', 0.0))
df_final['away_net_last5'] = df_final['away_team'].map(lambda t: forma.get(t, {}).get('net_last5', 0.0))
df_final['net_last5_diff'] = df_final['home_net_last5'] - df_final['away_net_last5']

FEATURES_BASE = [
    'elo_diff', 'overall_rating_diff', 'pace_diff',
    'home_gf_last5', 'home_gc_last5', 'away_gf_last5', 'away_gc_last5',
    'home_last5_winrate', 'away_last5_winrate',
    'h2h_last5_home_winrate', 'h2h_last5_avg_gd',
    'rank_diff', 'tier_diff',
    'fifa_rank_diff',         # lineal
    'fifa_rank_diff_log',     # logarítmica (NUEVA)
    'net_last5_diff'          # diferencia de neto (NUEVA)
]
df_final[FEATURES_BASE] = df_final[FEATURES_BASE].fillna(0)

print('Ejemplo de features (primeras 5):')
print(df_final[['home_team','away_team'] + FEATURES_BASE[:5]].head(3))

# ============================================================================
# CELDA 5 - Codificar neutral
# ============================================================================
df_final['neutral_num'] = df_final['neutral'].map(
    {True: 1, False: 0, 'True': 1, 'False': 0}
).fillna(0).astype(int)

print('Distribución neutral_num:')
print(df_final['neutral_num'].value_counts())

# ============================================================================
# CELDA 6 - Entrenamiento (MEJORADO)
# ============================================================================
FEATURES = [
    'elo_diff', 'overall_rating_diff', 'pace_diff',
    'home_gf_last5', 'home_gc_last5', 'away_gf_last5', 'away_gc_last5',
    'home_last5_winrate', 'away_last5_winrate',
    'h2h_last5_home_winrate', 'h2h_last5_avg_gd',
    'rank_diff', 'tier_diff',
    'fifa_rank_diff',        # lineal
    'fifa_rank_diff_log',    # logarítmica
    'net_last5_diff',        # diferencia de neto
    'neutral_num'
]

df_train = df_final.dropna(subset=['home_score', 'away_score'] + FEATURES).copy()

df_train['home_score_cls'] = df_train['home_score'].clip(upper=5).astype(int)
df_train['away_score_cls'] = df_train['away_score'].clip(upper=5).astype(int)

X      = df_train[FEATURES].values
y_home = df_train['home_score_cls'].values
y_away = df_train['away_score_cls'].values

X_train, X_test, yh_train, yh_test, ya_train, ya_test = train_test_split(
    X, y_home, y_away, test_size=0.2, random_state=42
)

print(f'Train: {X_train.shape[0]} partidos | Test: {X_test.shape[0]} partidos')
print(f'Features ({len(FEATURES)}): {FEATURES}')
print(f'Distribución goles local  : {pd.Series(y_home).value_counts().sort_index().to_dict()}')
print(f'Distribución goles visita : {pd.Series(y_away).value_counts().sort_index().to_dict()}')

# Ajuste de hiperparámetros para mejorar
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 15,
    'min_samples_leaf': 8,
    'random_state': 42,
    'n_jobs': -1
}

model_home = RandomForestClassifier(**RF_PARAMS)
model_away = RandomForestClassifier(**RF_PARAMS)

model_home.fit(X_train, yh_train)
model_away.fit(X_train, ya_train)

print('\n=== Clasificador Goles Local ===')
print(f'Precisión en test: {accuracy_score(yh_test, model_home.predict(X_test)):.3f}')
print('\n=== Clasificador Goles Visitante ===')
print(f'Precisión en test: {accuracy_score(ya_test, model_away.predict(X_test)):.3f}')

# ============================================================================
# CELDA 7 - Funciones auxiliares para simulación (ACTUALIZADAS)
# ============================================================================
def alinear_proba(proba, clases, k_range):
    vec = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        idx = np.where(clases == k)[0]
        if len(idx):
            vec[i] = proba[idx[0]]
    return vec

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

    h2h = df_final[
        (df_final['home_team'] == home_en) & (df_final['away_team'] == away_en)
    ].tail(5)
    h2h_wr = h2h['h2h_last5_home_winrate'].mean() if len(h2h) else 0.5
    h2h_gd = h2h['h2h_last5_avg_gd'].mean()       if len(h2h) else 0.0

    rank_h = rh['home_rank'].tail(1).values[0] if len(rh) else 50
    rank_a = ra['away_rank'].tail(1).values[0] if len(ra) else 50
    rank_diff_val = rank_h - rank_a

    tier_h = rh['home_rank_tier'].tail(1).values[0] if len(rh) else 2
    tier_a = ra['away_rank_tier'].tail(1).values[0] if len(ra) else 2
    tier_diff_val = tier_h - tier_a

    # Ranking FIFA (lineal y log)
    fifa_rank_h = get_rank(home_en)
    fifa_rank_a = get_rank(away_en)
    fifa_rank_diff = fifa_rank_h - fifa_rank_a
    fifa_rank_diff_log = np.sign(fifa_rank_diff) * np.log1p(np.abs(fifa_rank_diff))

    # Diferencia de neto en últimos 5
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

print("Funciones auxiliares cargadas.")

# ============================================================================
# CELDA 8 - Simulador con corrección exponencial (MEJORADO)
# ============================================================================
def simulador_partido_individual(equipo_1_es, equipo_2_es):
    """
    Predice el resultado con inferencia simétrica + corrección exponencial
    basada en ranking FIFA (factor 0.8).
    """
    clases_h  = model_home.classes_
    clases_a  = model_away.classes_
    max_goles = max(clases_h.max(), clases_a.max())
    k_range   = np.arange(0, max_goles + 1)

    # Inferencia simétrica
    X_s1 = get_features_partido(equipo_1_es, equipo_2_es, df_final, forma, neutral_val=1)
    p1_s1 = alinear_proba(model_home.predict_proba(X_s1)[0], clases_h, k_range)
    p2_s1 = alinear_proba(model_away.predict_proba(X_s1)[0], clases_a, k_range)

    X_s2 = get_features_partido(equipo_2_es, equipo_1_es, df_final, forma, neutral_val=1)
    p2_s2 = alinear_proba(model_home.predict_proba(X_s2)[0], clases_h, k_range)
    p1_s2 = alinear_proba(model_away.predict_proba(X_s2)[0], clases_a, k_range)

    p1_final = (p1_s1 + p1_s2) / 2.0
    p2_final = (p2_s1 + p2_s2) / 2.0

    g1_base = int(k_range[np.argmax(p1_final)])
    g2_base = int(k_range[np.argmax(p2_final)])

    # --- CORRECCIÓN EXPONENCIAL POST-PREDICCIÓN ---
    rank1 = get_rank(traducir(equipo_1_es))
    rank2 = get_rank(traducir(equipo_2_es))
    rank_diff = (rank2 - rank1) / 50.0   # positivo si equipo_1 es mejor
    correction = np.exp(rank_diff * 0.8) # factor 0.8 (ajusta según necesidad)

    g1 = max(0, int(round(g1_base * correction)))
    g2 = max(0, int(round(g2_base / correction if correction > 0 else g2_base)))

    # Si empate y gran diferencia de ranking, forzar un gol más al mejor
    if abs(rank_diff) > 0.3 and g1 == g2:
        if rank_diff > 0:
            g1 += 1
        else:
            g2 += 1

    g1 = min(g1, 7)
    g2 = min(g2, 7)

    # Determinar ganador y penales
    if g1 > g2:
        ganador = equipo_1_es
        penales = False
        pen_g1 = pen_g2 = None
    elif g2 > g1:
        ganador = equipo_2_es
        penales = False
        pen_g1 = pen_g2 = None
    else:
        penales = True
        prob_pen1 = 1 / (1 + np.exp(-rank_diff * 2.0))
        if random.random() < prob_pen1:
            pen_g1, pen_g2 = 5, 4
            ganador = equipo_1_es
        else:
            pen_g1, pen_g2 = 4, 5
            ganador = equipo_2_es

    return g1, g2, ganador, penales, pen_g1, pen_g2

# Prueba rápida
print(simulador_partido_individual('Francia', 'Argentina'))

# ============================================================================
# CELDA 9 - Exportar modelos y datos
# ============================================================================
with open('model_home.pkl', 'wb') as f:
    pickle.dump(model_home, f)
with open('model_away.pkl', 'wb') as f:
    pickle.dump(model_away, f)
with open('df_final.pkl', 'wb') as f:
    pickle.dump(df_final, f)
with open('forma.pkl', 'wb') as f:
    pickle.dump(forma, f)
with open('player_stats.pkl', 'wb') as f:
    pickle.dump(player_stats, f)

print("✅ Modelos y datos exportados correctamente.")