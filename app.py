from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
import random
import math

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ============================================================================
# CARGA DE MODELOS
# ============================================================================
try:
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
    MODELO_CARGADO = True
    print("✅ Modelos cargados correctamente.")
except Exception as e:
    print(f"⚠️  Error cargando modelos: {e}")
    MODELO_CARGADO = False

# ============================================================================
# NOTA IMPORTANTE
# ============================================================================
# Antes existía aquí un diccionario FIFA_RANKING escrito a mano (un único
# snapshot desactualizado del ranking), usado tanto como feature de
# entrenamiento como para una "corrección exponencial" manual aplicada
# después de cada predicción. Ese diccionario estaba desincronizado con el
# ranking real (ej. tenía a Croacia en el puesto 6 y a Ghana en el 43, cuando
# el propio dataset histórico los ubica en el 11 y el 76 respectivamente),
# lo que producía resultados sin sentido (equipos fuertes terminando último
# en la fase de grupos). Se eliminó por completo: ahora el ranking se lee
# siempre del histórico real (ver get_features_partido) y ya no se aplica
# ninguna corrección manual sobre la salida del modelo.
#
# ============================================================================
# DICCIONARIO ESPAÑOL → INGLÉS
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
    """Traduce un nombre de selección del español al inglés."""
    nombre_es = nombre_es.strip()
    if nombre_es in ES_TO_EN:
        return ES_TO_EN[nombre_es]
    for k, v in ES_TO_EN.items():
        if k.lower() == nombre_es.lower():
            return v
    return nombre_es  # fallback

# ============================================================================
# FUNCIONES DEL MODELO
# ============================================================================
def get_features_partido(home_es, away_es, df_final, forma, neutral_val=1):
    """Construye el vector de features con todas las características."""
    home_en = traducir(home_es)
    away_en = traducir(away_es)

    # ELO
    row_home = df_final[df_final['home_team'] == home_en].tail(1)
    row_away = df_final[df_final['away_team'] == away_en].tail(1)
    elo_home = row_home['elo_home_pre'].values[0] if len(row_home) else 1500.0
    elo_away = row_away['elo_away_pre'].values[0] if len(row_away) else 1500.0
    elo_diff = elo_home - elo_away

    # Jugadores (rating y pace)
    ph = player_stats[player_stats['team'] == home_en]
    pa = player_stats[player_stats['team'] == away_en]
    rating_home = ph['overall_rating_avg'].values[0] if len(ph) else 70.0
    rating_away = pa['overall_rating_avg'].values[0] if len(pa) else 70.0
    pace_home = ph['pace_avg'].values[0] if len(ph) else 70.0
    pace_away = pa['pace_avg'].values[0] if len(pa) else 70.0

    overall_rating_diff = rating_home - rating_away
    pace_diff_val = pace_home - pace_away

    # Forma reciente
    fh = forma.get(home_en, {'gf_last5': 1.2, 'gc_last5': 1.0, 'net_last5': 0.0})
    fa = forma.get(away_en, {'gf_last5': 1.2, 'gc_last5': 1.0, 'net_last5': 0.0})

    # Winrate últimos 5
    rh = df_final[df_final['home_team'] == home_en]
    ra = df_final[df_final['away_team'] == away_en]
    home_last5_wr = rh['home_last5_winrate'].tail(1).values[0] if len(rh) else 0.5
    away_last5_wr = ra['away_last5_winrate'].tail(1).values[0] if len(ra) else 0.5

    # Historial de enfrentamientos directos
    h2h = df_final[(df_final['home_team'] == home_en) & (df_final['away_team'] == away_en)].tail(5)
    h2h_wr = h2h['h2h_last5_home_winrate'].mean() if len(h2h) else 0.5
    h2h_gd = h2h['h2h_last5_avg_gd'].mean() if len(h2h) else 0.0

    # Rank y tier — ranking real (último valor observado en el histórico para
    # ese equipo), no un diccionario inventado.
    rank_h = rh['home_rank'].tail(1).values[0] if len(rh) else 50
    rank_a = ra['away_rank'].tail(1).values[0] if len(ra) else 50
    rank_diff_val = rank_h - rank_a
    rank_diff_log_val = np.sign(rank_diff_val) * np.log1p(np.abs(rank_diff_val))

    tier_h = rh['home_rank_tier'].tail(1).values[0] if len(rh) else 2
    tier_a = ra['away_rank_tier'].tail(1).values[0] if len(ra) else 2
    tier_diff_val = tier_h - tier_a

    # Diferencia de neto en últimos 5
    net_last5_diff = fh['net_last5'] - fa['net_last5']

    # Vector de features (16 características)
    feats = [
        elo_diff,
        overall_rating_diff,
        pace_diff_val,
        fh['gf_last5'],
        fh['gc_last5'],
        fa['gf_last5'],
        fa['gc_last5'],
        home_last5_wr,
        away_last5_wr,
        h2h_wr,
        h2h_gd,
        rank_diff_val,
        tier_diff_val,
        rank_diff_log_val,
        net_last5_diff,
        neutral_val
    ]
    return np.array(feats).reshape(1, -1)

# ============================================================================
# PREDICCIÓN (regresión de goles esperados, sin corrección manual)
# ============================================================================
def predecir_partido(equipo_1_es, equipo_2_es):
    if not MODELO_CARGADO:
        return _demo_partido(equipo_1_es, equipo_2_es)

    # Inferencia simétrica: se predice dos veces intercambiando quién figura
    # como local en el vector de features, y se promedian los goles esperados
    # (lambda) que el propio regresor entrega. Ya no existe ninguna
    # corrección manual posterior: el marcador sale directamente del modelo.
    X_s1 = get_features_partido(equipo_1_es, equipo_2_es, df_final, forma, neutral_val=1)
    g1_s1 = float(model_home.predict(X_s1)[0])
    g2_s1 = float(model_away.predict(X_s1)[0])

    X_s2 = get_features_partido(equipo_2_es, equipo_1_es, df_final, forma, neutral_val=1)
    g2_s2 = float(model_home.predict(X_s2)[0])
    g1_s2 = float(model_away.predict(X_s2)[0])

    lam1 = max(0.05, (g1_s1 + g1_s2) / 2.0)
    lam2 = max(0.05, (g2_s1 + g2_s2) / 2.0)

    # El marcador final se muestrea de una distribución de Poisson centrada
    # en los goles esperados (técnica estándar para simular partidos a partir
    # de un modelo de goles esperados). Redondear el promedio directamente
    # producía demasiados empates artificiales cuando dos equipos tenían
    # valores esperados muy parecidos, aplanando diferencias reales de nivel.
    g1 = int(np.random.poisson(lam1))
    g2 = int(np.random.poisson(lam2))

    # Penales: si el resultado queda empatado, el ganador se decide con la
    # diferencia de goles esperados (lam1 - lam2) que ya predijo el modelo,
    # no con una tabla de ranking externa.
    penales = False
    pen_g1 = pen_g2 = None
    if g1 == g2:
        penales = True
        prob_pen1 = 1 / (1 + np.exp(-(lam1 - lam2) * 2.0))
        if random.random() < prob_pen1:
            pen_g1 = random.randint(4, 5)
            pen_g2 = random.randint(2, 4)
            ganador = equipo_1_es
        else:
            pen_g1 = random.randint(2, 4)
            pen_g2 = random.randint(4, 5)
            ganador = equipo_2_es
    else:
        ganador = equipo_1_es if g1 > g2 else equipo_2_es

    return {
        "equipo1": equipo_1_es,
        "equipo2": equipo_2_es,
        "g1": g1,
        "g2": g2,
        "penales": penales,
        "pen_g1": pen_g1,
        "pen_g2": pen_g2,
        "ganador": ganador,
        "probs1": _poisson_pmf(lam1, 6),
        "probs2": _poisson_pmf(lam2, 6),
    }

def _poisson_pmf(lam, n):
    """Distribución de probabilidad de goles (0..n-1) a partir del valor
    esperado (lambda) que predice el modelo, usando la distribución de
    Poisson (estándar en modelos de fútbol) para alimentar las barras de
    probabilidad de la interfaz."""
    lam = max(lam, 0.01)
    probs = [np.exp(-lam) * lam ** k / math.factorial(k) for k in range(n)]
    total = sum(probs)
    return [round(p / total * 100, 1) for p in probs]

def _demo_partido(eq1, eq2):
    """Modo demo cuando no hay modelos cargados (sin datos de ninguna
    selección disponibles, por lo que no se favorece a ningún equipo)."""
    g1 = int(np.random.poisson(1.3))
    g2 = int(np.random.poisson(1.3))
    penales = False
    pen_g1 = pen_g2 = None
    if g1 == g2:
        penales = True
        if random.random() > 0.5:
            pen_g1, pen_g2 = 5, random.randint(2, 4)
            ganador = eq1
        else:
            pen_g1, pen_g2 = random.randint(2, 4), 5
            ganador = eq2
    else:
        ganador = eq1 if g1 > g2 else eq2
    return {
        "equipo1": eq1,
        "equipo2": eq2,
        "g1": g1,
        "g2": g2,
        "penales": penales,
        "pen_g1": pen_g1,
        "pen_g2": pen_g2,
        "ganador": ganador,
        "probs1": _poisson_pmf(1.3, 6),
        "probs2": _poisson_pmf(1.3, 6),
        "demo": True,
    }

# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================
@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    eq1 = data.get('equipo1', '').strip()
    eq2 = data.get('equipo2', '').strip()
    if not eq1 or not eq2:
        return jsonify({"error": "Faltan equipos"}), 400
    try:
        return jsonify(predecir_partido(eq1, eq2))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/paises', methods=['GET'])
def paises():
    """Devuelve la lista de países en español para el autocompletado."""
    return jsonify(sorted(ES_TO_EN.keys()))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True, "modelo_cargado": MODELO_CARGADO})

if __name__ == '__main__':
    app.run(debug=True, port=5000)