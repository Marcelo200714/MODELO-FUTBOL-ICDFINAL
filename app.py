from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
import random

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CARGA DE MODELOS
# ============================================================================
try:
    with open(os.path.join(BASE_DIR, 'model_home.pkl'), 'rb') as f:
        model_home = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model_away.pkl'), 'rb') as f:
        model_away = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'df_final.pkl'), 'rb') as f:
        df_final = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'forma.pkl'), 'rb') as f:
        forma = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'player_stats.pkl'), 'rb') as f:
        player_stats = pickle.load(f)
    MODELO_CARGADO = True
    print("✅ Modelos cargados correctamente.")
except Exception as e:
    print(f"⚠️  Error cargando modelos: {e}")
    MODELO_CARGADO = False

# ============================================================================
# RANKING FIFA (mismo que el notebook)
# ============================================================================
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
    """Devuelve el ranking FIFA de un equipo (en inglés)."""
    return FIFA_RANKING.get(team, 100)

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
    """Traduce un nombre de selección del español al inglés."""
    nombre_es = nombre_es.strip()
    if nombre_es in ES_TO_EN:
        return ES_TO_EN[nombre_es]
    for k, v in ES_TO_EN.items():
        if k.lower() == nombre_es.lower():
            return v
    return nombre_es  # fallback

# ============================================================================
# FUNCIONES DEL MODELO (con todas las nuevas características)
# ============================================================================
def alinear_proba(proba, clases, k_range):
    vec = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        idx = np.where(clases == k)[0]
        if len(idx):
            vec[i] = proba[idx[0]]
    return vec

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

    # Rank y tier
    rank_h = rh['home_rank'].tail(1).values[0] if len(rh) else 50
    rank_a = ra['away_rank'].tail(1).values[0] if len(ra) else 50
    rank_diff_val = rank_h - rank_a

    tier_h = rh['home_rank_tier'].tail(1).values[0] if len(rh) else 2
    tier_a = ra['away_rank_tier'].tail(1).values[0] if len(ra) else 2
    tier_diff_val = tier_h - tier_a

    # Ranking FIFA (lineal y logarítmica)
    fifa_rank_h = get_rank(home_en)
    fifa_rank_a = get_rank(away_en)
    fifa_rank_diff = fifa_rank_h - fifa_rank_a
    fifa_rank_diff_log = np.sign(fifa_rank_diff) * np.log1p(np.abs(fifa_rank_diff))

    # Diferencia de neto en últimos 5
    net_last5_diff = fh['net_last5'] - fa['net_last5']

    # Vector de features (17 características)
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
        fifa_rank_diff,
        fifa_rank_diff_log,
        net_last5_diff,
        neutral_val
    ]
    return np.array(feats).reshape(1, -1)

# ============================================================================
# PREDICCIÓN CON CORRECCIÓN EXPONENCIAL (factor 0.8)
# ============================================================================
def predecir_partido(equipo_1_es, equipo_2_es):
    if not MODELO_CARGADO:
        return _demo_partido(equipo_1_es, equipo_2_es)

    eq1_en = traducir(equipo_1_es)
    eq2_en = traducir(equipo_2_es)
    rank1 = get_rank(eq1_en)
    rank2 = get_rank(eq2_en)

    clases_h = model_home.classes_
    clases_a = model_away.classes_
    max_goles = max(clases_h.max(), clases_a.max())
    k_range = np.arange(0, max_goles + 1)

    # Inferencia simétrica (promedia dos escenarios)
    X_s1 = get_features_partido(equipo_1_es, equipo_2_es, df_final, forma, neutral_val=1)
    p1_s1 = alinear_proba(model_home.predict_proba(X_s1)[0], clases_h, k_range)
    p2_s1 = alinear_proba(model_away.predict_proba(X_s1)[0], clases_a, k_range)

    X_s2 = get_features_partido(equipo_2_es, equipo_1_es, df_final, forma, neutral_val=1)
    p2_s2 = alinear_proba(model_home.predict_proba(X_s2)[0], clases_h, k_range)
    p1_s2 = alinear_proba(model_away.predict_proba(X_s2)[0], clases_a, k_range)

    p1_final = (p1_s1 + p1_s2) / 2.0
    p2_final = (p2_s1 + p2_s2) / 2.0

    # Goles base (sin corregir)
    g1_base = int(k_range[np.argmax(p1_final)])
    g2_base = int(k_range[np.argmax(p2_final)])

    # --- CORRECCIÓN EXPONENCIAL (factor 0.8) ---
    rank_diff = (rank2 - rank1) / 50.0  # positivo si eq1 mejor
    correction = np.exp(rank_diff * 0.8)

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

    # Penales
    penales = False
    pen_g1 = pen_g2 = None
    if g1 == g2:
        penales = True
        prob_pen1 = 1 / (1 + np.exp(-rank_diff * 2.0))
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
        "probs1": [round(float(p) * 100, 1) for p in p1_final[:6]],
        "probs2": [round(float(p) * 100, 1) for p in p2_final[:6]],
    }

def _demo_partido(eq1, eq2):
    """Modo demo cuando no hay modelos cargados."""
    eq1_en = traducir(eq1)
    eq2_en = traducir(eq2)
    rank1 = get_rank(eq1_en)
    rank2 = get_rank(eq2_en)
    rank_diff = (rank2 - rank1) / 50.0
    fuerza1 = max(1, 100 - rank1)
    fuerza2 = max(1, 100 - rank2)
    media1 = 1.2 + (fuerza1 / (fuerza1 + fuerza2)) * 1.8
    media2 = 1.2 + (fuerza2 / (fuerza1 + fuerza2)) * 1.8
    g1 = min(6, np.random.poisson(media1))
    g2 = min(6, np.random.poisson(media2))
    if g1 == g2:
        if rank_diff > 0:
            g1 += 1
        else:
            g2 += 1
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
        "probs1": [25, 34, 25, 10, 4, 2],
        "probs2": [30, 33, 22, 10, 3, 2],
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