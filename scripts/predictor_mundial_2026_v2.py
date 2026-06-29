# ============================================================================
# CELDA 1 - Librerías y carga
# ============================================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import random
import os

# Configura rutas (relativas a la carpeta raíz del proyecto)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

PATH_MATCHES = os.path.join(DATA_DIR, 'national_matches_(1992-2026).csv')
PATH_PLAYERS = os.path.join(DATA_DIR, 'player-data-full.csv')
PATH_RESULTS = os.path.join(DATA_DIR, 'results.csv')
PATH_WC      = os.path.join(DATA_DIR, 'worldcup_matches.csv')

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
# CELDA 4 - Feature engineering (CORREGIDO)
# ============================================================================
# El dataset YA trae 'home_rank' / 'away_rank' / 'rank_diff' / 'tier_diff' reales
# y variables en el tiempo (ranking FIFA real de cada equipo en la fecha de cada
# partido). La versión anterior de este script además usaba un diccionario
# FIFA_RANKING escrito a mano con un único snapshot de ranking, desactualizado
# e inconsistente con los datos reales (ej. Croacia figuraba en el puesto 6
# cuando el propio dataset la ubica en el puesto 11; Ghana figuraba en el 43
# cuando en realidad está en el 76). Esa feature inventada ('fifa_rank_diff')
# se entrenaba en paralelo con la real ('rank_diff'), dándole al modelo dos
# señales contradictorias del mismo concepto y degradando sus predicciones
# para selecciones fuertes cuyo ranking real no coincidía con el inventado.
#
# Se elimina el diccionario y se deriva la transformación logarítmica
# directamente de la columna real 'rank_diff'.
df_final['overall_rating_diff'] = (
    df_final['home_overall_rating'].fillna(70) - df_final['away_overall_rating'].fillna(70)
)
df_final['pace_diff'] = (
    df_final['home_pace'].fillna(70) - df_final['away_pace'].fillna(70)
)

# Transformación logarítmica del ranking real (capta mejor la no linealidad:
# equipos muy superiores tienen ventaja extra) sin depender de datos inventados.
df_final['rank_diff_log'] = np.sign(df_final['rank_diff']) * np.log1p(np.abs(df_final['rank_diff']))

# Forma reciente (últimos 5 partidos)
#
# BUG DE FONDO: promediar goles crudos de los últimos 5 partidos no ajusta
# por la fuerza del rival. Selecciones que golean a rivales débiles de su
# propia confederación (ej. una goleada 8-0 en clasificatorias de Oceanía u
# CONCACAF) terminan con una "forma reciente" inflada muy por encima de
# selecciones top que juegan calendarios mucho más parejos (Europa/Sudamérica),
# aunque su nivel real sea muy inferior. Esa señal ruidosa competía en el
# modelo con el ranking real y podía revertir el resultado esperado en
# cruces claros (ej. una potencia vs. una selección menor). Se acota cada
# partido individual a un máximo de 4 goles a favor/en contra antes de
# promediar, para que una goleada puntual no pese desproporcionadamente más
# que una victoria ajustada ante un rival de nivel similar.
GOLEADA_MAX = 4

def calcular_forma(df_results, ventana=5):
    df_r = df_results.sort_values('date').copy()
    forma_dict = {}
    todos_equipos = pd.concat([df_r['home_team'], df_r['away_team']]).unique()
    for equipo in todos_equipos:
        como_local = df_r[df_r['home_team'] == equipo][['date','home_score','away_score']].rename(
            columns={'home_score':'gf','away_score':'gc'})
        como_visitante = df_r[df_r['away_team'] == equipo][['date','away_score','home_score']].rename(
            columns={'away_score':'gf','home_score':'gc'})
        partidos = pd.concat([como_local, como_visitante]).sort_values('date').tail(ventana).copy()
        partidos['gf'] = partidos['gf'].clip(upper=GOLEADA_MAX)
        partidos['gc'] = partidos['gc'].clip(upper=GOLEADA_MAX)
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

df_final['home_net_last5'] = df_final['home_team'].map(lambda t: forma.get(t, {}).get('net_last5', 0.0))
df_final['away_net_last5'] = df_final['away_team'].map(lambda t: forma.get(t, {}).get('net_last5', 0.0))
df_final['net_last5_diff'] = df_final['home_net_last5'] - df_final['away_net_last5']

FEATURES_BASE = [
    'elo_diff', 'overall_rating_diff', 'pace_diff',
    'home_gf_last5', 'home_gc_last5', 'away_gf_last5', 'away_gc_last5',
    'home_last5_winrate', 'away_last5_winrate',
    'h2h_last5_home_winrate', 'h2h_last5_avg_gd',
    'rank_diff', 'tier_diff',
    'rank_diff_log',    # log del ranking REAL (antes usaba un ranking inventado)
    'net_last5_diff'
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
# CELDA 6 - Entrenamiento (RandomForestRegressor + búsqueda de hiperparámetros)
# ============================================================================
# Antes se entrenaba un RandomForestClassifier prediciendo la clase de goles
# más probable (con clip(upper=5)). Un clasificador que solo devuelve la moda
# de la distribución "aplana" la diferencia entre equipos: dos selecciones muy
# distintas en nivel terminan prediciendo el mismo marcador porque la clase
# más probable para ambas es igual (0 o 1 gol), y esa pérdida de información
# es la que después se intentaba "arreglar" a mano con una corrección
# exponencial basada en el ranking inventado.
#
# Ahora se usa RandomForestRegressor: predice directamente el número esperado
# de goles (continuo), por lo que cualquier diferencia de nivel entre equipos
# se refleja de forma suave y directa en la salida del propio modelo, sin
# necesitar ninguna corrección manual posterior.
FEATURES = [
    'elo_diff', 'overall_rating_diff', 'pace_diff',
    'home_gf_last5', 'home_gc_last5', 'away_gf_last5', 'away_gc_last5',
    'home_last5_winrate', 'away_last5_winrate',
    'h2h_last5_home_winrate', 'h2h_last5_avg_gd',
    'rank_diff', 'tier_diff',
    'rank_diff_log',
    'net_last5_diff',
    'neutral_num'
]

df_train = df_final.dropna(subset=['home_score', 'away_score'] + FEATURES).copy()

# Solo se acota goleadas extremas (blowouts de 15-31 goles en clasificatorias
# contra selecciones amateur) para que no dominen la varianza de los árboles;
# a diferencia del clip(upper=5) anterior, esto conserva diferenciación real
# entre un resultado de 3, 5 u 8 goles.
df_train['home_score_reg'] = df_train['home_score'].clip(upper=8)
df_train['away_score_reg'] = df_train['away_score'].clip(upper=8)

X      = df_train[FEATURES].values
y_home = df_train['home_score_reg'].values
y_away = df_train['away_score_reg'].values

X_train, X_test, yh_train, yh_test, ya_train, ya_test = train_test_split(
    X, y_home, y_away, test_size=0.2, random_state=42
)

print(f'Train: {X_train.shape[0]} partidos | Test: {X_test.shape[0]} partidos')
print(f'Features ({len(FEATURES)}): {FEATURES}')

# Búsqueda de hiperparámetros por validación cruzada (en vez de valores fijos
# elegidos a mano): se deja que el propio proceso de selección de modelo
# encuentre la combinación que minimiza el error, con 3 folds de CV.
# max_depth y min_samples_leaf se acotan a un rango razonable (nada de
# profundidad ilimitada) para no generar árboles innecesariamente grandes:
# un bosque más profundo de lo necesario no mejora la generalización y solo
# infla el tamaño del .pkl exportado.
PARAM_DIST = {
    'n_estimators': [150, 200, 300, 400],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [4, 8, 12, 16],
    'max_features': ['sqrt', 'log2', 0.5],
}

def buscar_mejor_modelo(X_tr, y_tr, nombre):
    base = RandomForestRegressor(random_state=42, n_jobs=1)
    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=15, cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_tr, y_tr)
    print(f'\nMejores hiperparámetros ({nombre}): {search.best_params_}')
    print(f'MAE promedio en CV ({nombre}): {-search.best_score_:.3f} goles')
    return search.best_estimator_

model_home = buscar_mejor_modelo(X_train, yh_train, 'goles local')
model_away = buscar_mejor_modelo(X_train, ya_train, 'goles visitante')

pred_h = model_home.predict(X_test)
pred_a = model_away.predict(X_test)

print('\n=== Regresor Goles Local ===')
print(f'MAE en test : {mean_absolute_error(yh_test, pred_h):.3f} goles')
print(f'RMSE en test: {mean_squared_error(yh_test, pred_h) ** 0.5:.3f} goles')
print('\n=== Regresor Goles Visitante ===')
print(f'MAE en test : {mean_absolute_error(ya_test, pred_a):.3f} goles')
print(f'RMSE en test: {mean_squared_error(ya_test, pred_a) ** 0.5:.3f} goles')

print('\nImportancia de features (goles local):')
for feat, imp in sorted(zip(FEATURES, model_home.feature_importances_), key=lambda x: -x[1]):
    print(f'  {feat:30s} {imp:.3f}')

# ============================================================================
# CELDA 7 - Funciones auxiliares para simulación
# ============================================================================
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

    # Ranking real (último valor observado para ese equipo en el dataset),
    # NO un diccionario inventado.
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

print("Funciones auxiliares cargadas.")

# ============================================================================
# CELDA 8 - Simulador con inferencia simétrica (sin corrección manual)
# ============================================================================
def simulador_partido_individual(equipo_1_es, equipo_2_es):
    """
    Predice el resultado promediando dos inferencias simétricas (intercambiando
    quién figura como local en el vector de features), tal como antes, pero
    ya no aplica ninguna corrección manual posterior: el marcador sale
    directamente de lo que el modelo entrenado predice.
    """
    X_s1 = get_features_partido(equipo_1_es, equipo_2_es, df_final, forma, neutral_val=1)
    g1_s1 = model_home.predict(X_s1)[0]
    g2_s1 = model_away.predict(X_s1)[0]

    X_s2 = get_features_partido(equipo_2_es, equipo_1_es, df_final, forma, neutral_val=1)
    g2_s2 = model_home.predict(X_s2)[0]
    g1_s2 = model_away.predict(X_s2)[0]

    lam1 = (g1_s1 + g1_s2) / 2.0
    lam2 = (g2_s1 + g2_s2) / 2.0

    g1 = max(0, int(round(lam1)))
    g2 = max(0, int(round(lam2)))

    if g1 > g2:
        ganador = equipo_1_es
        penales = False
        pen_g1 = pen_g2 = None
    elif g2 > g1:
        ganador = equipo_2_es
        penales = False
        pen_g1 = pen_g2 = None
    else:
        # Empate: se define el ganador de penales con la diferencia de goles
        # esperados (lam1 - lam2) que el propio modelo ya predijo, no con un
        # ranking externo.
        penales = True
        prob_pen1 = 1 / (1 + np.exp(-(lam1 - lam2) * 2.0))
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
os.makedirs(MODELS_DIR, exist_ok=True)
with open(os.path.join(MODELS_DIR, 'model_home.pkl'), 'wb') as f:
    pickle.dump(model_home, f)
with open(os.path.join(MODELS_DIR, 'model_away.pkl'), 'wb') as f:
    pickle.dump(model_away, f)
with open(os.path.join(MODELS_DIR, 'df_final.pkl'), 'wb') as f:
    pickle.dump(df_final, f)
with open(os.path.join(MODELS_DIR, 'forma.pkl'), 'wb') as f:
    pickle.dump(forma, f)
with open(os.path.join(MODELS_DIR, 'player_stats.pkl'), 'wb') as f:
    pickle.dump(player_stats, f)

print("✅ Modelos y datos exportados correctamente.")
