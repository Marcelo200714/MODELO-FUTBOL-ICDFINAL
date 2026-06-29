import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import os
import math
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

# --------------------------------------------
# 2. Bracket real de dieciseisavos (Mundial 2026)
# --------------------------------------------
# Los 16 cruces tal como quedó el cuadro de dieciseisavos de final una vez
# cerrada la fase de grupos (32 clasificados reales), no un bracket de
# ejemplo. Los nombres usan la grafía exacta del dataset (national_matches),
# igual que en generar_fase_grupos.py.
PARTIDOS_16 = [
    ('Canada', 'South Africa'),
    ('Paraguay', 'Germany'),
    ('Morocco', 'Netherlands'),
    ('Brazil', 'Japan'),
    ('France', 'Sweden'),
    ('Norway', "Côte d'Ivoire"),
    ('Mexico', 'Ecuador'),
    ('England', 'Congo DR'),
    ('USA', 'Bosnia and Herzegovina'),
    ('Belgium', 'Senegal'),
    ('Portugal', 'Croatia'),
    ('Spain', 'Austria'),
    ('Switzerland', 'Algeria'),
    ('Argentina', 'Cabo Verde'),
    ('Colombia', 'Ghana'),
    ('Egypt', 'Australia'),
]

# --------------------------------------------
# 3. Función de features (idéntica a generar_fase_grupos.py)
# --------------------------------------------
def traducir(nombre_es):
    return nombre_es  # los cruces ya están en la grafía del dataset

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
    return np.array(feats).reshape(1, -1), int(rank_h), int(rank_a)

# --------------------------------------------
# 4. Predicción determinista (Poisson exacto, sin aleatoriedad),
#    igual que en generar_fase_grupos.py, pero adaptada a eliminación
#    directa: en dieciseisavos no puede haber empate, así que si el modelo
#    da el empate como desenlace más probable, el cruce se resuelve por
#    penales tomando como favorito al equipo con mayor gol esperado (lambda).
# --------------------------------------------
def _poisson_pmf(lam, max_goals):
    return [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(max_goals + 1)]

def match_probs(lam1, lam2, max_goals=10):
    p1 = _poisson_pmf(lam1, max_goals)
    p2 = _poisson_pmf(lam2, max_goals)
    p_home_win = sum(p1[i] * p2[j] for i in range(max_goals + 1) for j in range(i))
    p_draw     = sum(p1[i] * p2[i] for i in range(max_goals + 1))
    p_away_win = sum(p1[i] * p2[j] for i in range(max_goals + 1) for j in range(i + 1, max_goals + 1))
    return p_home_win, p_draw, p_away_win

def predict_knockout(home_es, away_es):
    X_s1, rank_h, rank_a = get_features_partido(home_es, away_es, df_final, forma, neutral_val=1)
    g1_s1 = float(model_home.predict(X_s1)[0])
    g2_s1 = float(model_away.predict(X_s1)[0])

    X_s2, _, _ = get_features_partido(away_es, home_es, df_final, forma, neutral_val=1)
    g2_s2 = float(model_home.predict(X_s2)[0])
    g1_s2 = float(model_away.predict(X_s2)[0])

    lam1 = max(0.05, (g1_s1 + g1_s2) / 2.0)
    lam2 = max(0.05, (g2_s1 + g2_s2) / 2.0)

    p_home, p_draw, p_away = match_probs(lam1, lam2)

    gh, ga = round(lam1), round(lam2)
    penales = False
    if p_home >= p_away and p_home >= p_draw:
        if gh <= ga:
            gh = ga + 1
        ganador = home_es
    elif p_away > p_home and p_away >= p_draw:
        if ga <= gh:
            ga = gh + 1
        ganador = away_es
    else:
        gh = ga = round((lam1 + lam2) / 2.0)
        penales = True
        ganador = home_es if lam1 >= lam2 else away_es

    return {
        'home': home_es, 'away': away_es, 'gh': gh, 'ga': ga,
        'rank_h': rank_h, 'rank_a': rank_a,
        'penales': penales, 'ganador': ganador,
    }

# --------------------------------------------
# 5. Simular dieciseisavos completos
# --------------------------------------------
resultados = [predict_knockout(home, away) for home, away in PARTIDOS_16]

print("\n--- DIECISEISAVOS DE FINAL: PREDICCIÓN DEL MODELO ---")
for r in resultados:
    pen = '  (definido por penales)' if r['penales'] else ''
    print(f"{r['home']:25s} {r['gh']} - {r['ga']} {r['away']:25s} -> avanza: {r['ganador']}{pen}")

# --------------------------------------------
# 6. Visualizar bracket (4x4) con marcador y ganador resaltado
# --------------------------------------------
def show_r32_predictions(resultados):
    n = len(resultados)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.2 * n_rows))
    fig.suptitle('DIECISEISAVOS DE FINAL — PREDICCIÓN DEL MODELO (bracket real)', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i in range(n, len(axes)):
        axes[i].axis('off')

    for idx, r in enumerate(resultados):
        ax = axes[idx]
        ax.axis('off')
        ax.text(0.5, 0.95, f"Cruce {idx+1}", transform=ax.transAxes,
                fontsize=10, ha='center', va='center', color='gray')

        home_win = r['ganador'] == r['home']
        color_h = 'green' if home_win else 'black'
        color_a = 'green' if not home_win else 'black'
        weight_h = 'bold' if home_win else 'normal'
        weight_a = 'bold' if not home_win else 'normal'

        ax.text(0.5, 0.68, f"{r['home']} ({r['gh']})", transform=ax.transAxes,
                fontsize=12, ha='center', va='center', color=color_h, fontweight=weight_h)
        ax.text(0.5, 0.52, 'vs', transform=ax.transAxes,
                fontsize=9, ha='center', va='center', color='gray')
        ax.text(0.5, 0.36, f"{r['away']} ({r['ga']})", transform=ax.transAxes,
                fontsize=12, ha='center', va='center', color=color_a, fontweight=weight_a)

        if r['penales']:
            ax.text(0.5, 0.16, 'definido por penales', transform=ax.transAxes,
                    fontsize=8, ha='center', va='center', color='#e67e22')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

show_r32_predictions(resultados)

# --------------------------------------------
# 7. Clasificados a octavos de final
# --------------------------------------------
print("\n--- CLASIFICADOS A OCTAVOS DE FINAL ---")
for r in resultados:
    print(f"  {r['ganador']}")
