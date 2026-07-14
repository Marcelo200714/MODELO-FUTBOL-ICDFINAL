# ⚽ Predictor Mundial 2026 — Guía de uso

Este repositorio contiene un sistema de predicción de resultados de la Copa Mundial de la FIFA 2026 basado en RandomForest, con interfaz web y generación automática de la fase de grupos.

## 📁 Estructura del proyecto

```
.
├── app.py                     # API Flask (servidor de predicciones)
├── index.html                 # Interfaz web (bracket + predictor)
├── requirements.txt
├── data/
│   └── raw/                   # CSVs de origen (partidos, jugadores, resultados)
├── models/                    # Modelos entrenados y datos procesados (.pkl)
└── scripts/
    ├── predictor_mundial_2026_v2.py   # Entrenamiento del modelo
    ├── generar_fase_grupos.py         # Generación de la fase de grupos
    └── exportar_modelos.py            # Utilidad para exportar desde un notebook
```

## 📊 Resultados predichos: Fase de grupos (Mundial 2026)

Estas son las tablas de posiciones que arroja el modelo para los 12 grupos de la fase de grupos (formato de 48 selecciones), generadas con `scripts/generar_fase_grupos.py` (simulación determinista, sin números aleatorios: el mismo cruce de equipos siempre produce la misma tabla). En el formato real del Mundial 2026, avanzan a dieciseisavos los dos primeros de cada grupo más los 8 mejores terceros; esta tabla solo calcula la tabla de posiciones dentro de cada grupo, no ese desempate entre terceros. El bracket real de dieciseisavos de la sección siguiente ya incorpora el resultado final de esa clasificación y no se deriva automáticamente de esta simulación.

| Grupo | Pos | Selección | Pts | DG | GF |
|:-----:|:---:|-----------|:---:|:--:|:--:|
| A | 1 | **Mexico** | 9 | +3 | 6 |
| A | 2 | **South Africa** | 6 | +1 | 5 |
| A | 3 | South Korea | 3 | -1 | 4 |
| A | 4 | Czech Republic | 0 | -3 | 3 |
| B | 1 | **Switzerland** | 9 | +3 | 6 |
| B | 2 | **Canada** | 6 | +1 | 5 |
| B | 3 | Bosnia and Herzegovina | 3 | -1 | 4 |
| B | 4 | Qatar | 0 | -3 | 3 |
| C | 1 | **Brazil** | 9 | +3 | 6 |
| C | 2 | **Morocco** | 6 | +1 | 5 |
| C | 3 | Scotland | 3 | -1 | 4 |
| C | 4 | Haiti | 0 | -3 | 3 |
| D | 1 | **Paraguay** | 9 | +3 | 6 |
| D | 2 | **Australia** | 6 | +1 | 5 |
| D | 3 | USA | 3 | -1 | 4 |
| D | 4 | Turkey | 0 | -3 | 3 |
| E | 1 | **Germany** | 9 | +3 | 6 |
| E | 2 | **Ecuador** | 6 | +1 | 5 |
| E | 3 | Côte d'Ivoire | 3 | -1 | 4 |
| E | 4 | Curaçao | 0 | -3 | 3 |
| F | 1 | **Netherlands** | 9 | +3 | 6 |
| F | 2 | **Japan** | 6 | +1 | 5 |
| F | 3 | Sweden | 3 | -1 | 4 |
| F | 4 | Tunisia | 0 | -3 | 3 |
| G | 1 | **Belgium** | 9 | +3 | 6 |
| G | 2 | **Egypt** | 6 | +1 | 5 |
| G | 3 | Iran | 3 | -1 | 4 |
| G | 4 | New Zealand | 0 | -3 | 3 |
| H | 1 | **Spain** | 9 | +3 | 6 |
| H | 2 | **Uruguay** | 6 | +1 | 5 |
| H | 3 | Saudi Arabia | 3 | -1 | 4 |
| H | 4 | Cabo Verde | 0 | -3 | 3 |
| I | 1 | **France** | 9 | +3 | 6 |
| I | 2 | **Norway** | 6 | +1 | 5 |
| I | 3 | Senegal | 3 | -1 | 4 |
| I | 4 | Iraq | 0 | -3 | 3 |
| J | 1 | **Argentina** | 9 | +3 | 6 |
| J | 2 | **Austria** | 6 | +1 | 5 |
| J | 3 | Algeria | 3 | -1 | 4 |
| J | 4 | Jordan | 0 | -3 | 3 |
| K | 1 | **Portugal** | 9 | +3 | 6 |
| K | 2 | **Colombia** | 6 | +1 | 5 |
| K | 3 | Uzbekistan | 3 | -1 | 4 |
| K | 4 | Congo DR | 0 | -3 | 3 |
| L | 1 | **England** | 9 | +3 | 6 |
| L | 2 | **Croatia** | 6 | +1 | 5 |
| L | 3 | Panama | 3 | -1 | 4 |
| L | 4 | Ghana | 0 | -3 | 3 |

## 🏆 Resultados predichos: Dieciseisavos de final (Mundial 2026)

Estos son los resultados que arroja el modelo para el cuadro real de dieciseisavos de final (32 clasificados reales de la fase de grupos), generados con `scripts/generar_dieciseisavos.py`. Se incluyen aquí directamente para poder revisarlos sin necesidad de clonar el repositorio ni ejecutar el código.

| # | Local | Marcador | Visitante | Avanza a octavos |
|---|-------|:--------:|-----------|:----------------:|
| 1 | Canada | 2 - 1 | South Africa | **Canada** |
| 2 | Paraguay | 1 - 2 | Germany | **Germany** |
| 3 | Morocco | 1 - 2 | Netherlands | **Netherlands** |
| 4 | Brazil | 2 - 1 | Japan | **Brazil** |
| 5 | France | 2 - 1 | Sweden | **France** |
| 6 | Norway | 2 - 1 | Côte d'Ivoire | **Norway** |
| 7 | Mexico | 1 - 2 | Ecuador | **Ecuador** |
| 8 | England | 2 - 1 | Congo DR | **England** |
| 9 | USA | 2 - 1 | Bosnia and Herzegovina | **USA** |
| 10 | Belgium | 1 - 2 | Senegal | **Senegal** |
| 11 | Portugal | 2 - 1 | Croatia | **Portugal** |
| 12 | Spain | 2 - 1 | Austria | **Spain** |
| 13 | Switzerland | 2 - 1 | Algeria | **Switzerland** |
| 14 | Argentina | 2 - 1 | Cabo Verde | **Argentina** |
| 15 | Colombia | 2 - 1 | Ghana | **Colombia** |
| 16 | Egypt | 1 - 2 | Australia | **Australia** |

**Clasificados a octavos de final según el modelo:** Canada, Germany, Netherlands, Brazil, France, Norway, Ecuador, England, USA, Senegal, Portugal, Spain, Switzerland, Argentina, Colombia, Australia.

## 📋 Requisitos previos

- Python 3.9 o superior instalado.
- Gestor de paquetes `pip` disponible.
- Opcional: entorno virtual (recomendado).

## 🔧 Instalación de dependencias

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

Si no tienes el archivo `requirements.txt`, instala manualmente:

```bash
pip install pandas numpy scikit-learn matplotlib flask flask-cors
```

## 🧠 Paso 1: Entrenar el modelo (si no tienes los archivos .pkl)

El proyecto necesita los modelos entrenados (`models/model_home.pkl`, `models/model_away.pkl`) y los datos procesados (`models/df_final.pkl`, `models/forma.pkl`, `models/player_stats.pkl`).
Si no los tienes, ejecuta el script de entrenamiento desde la raíz del proyecto:

```bash
python scripts/predictor_mundial_2026_v2.py
```

Este script:

- Carga los CSVs desde `data/raw/` (`national_matches_(1992-2026).csv`, `player-data-full.csv`, `results.csv`, `worldcup_matches.csv`).
- Realiza feature engineering (incluye ranking FIFA, forma reciente, etc.).
- Entrena dos `RandomForestClassifier` (goles local y visitante).
- Guarda los archivos `.pkl` necesarios en `models/`.

Salida esperada: verás en consola la precisión del modelo y el mensaje "✅ Modelos y datos exportados correctamente."

## 🖼️ Paso 2: Generar la fase de grupos (imagen 4×3)

Una vez entrenado el modelo, puedes generar las tablas de la fase de grupos con los puntajes predichos, ejecutando desde la raíz del proyecto:

```bash
python scripts/generar_fase_grupos.py
```

Esto:

- Lee los partidos de la fase de grupos del Mundial 2026 desde `data/raw/worldcup_matches.csv`.
- Predice cada partido usando el modelo (con corrección exponencial por ranking).
- Calcula puntos, diferencia de goles y goles a favor.
- Muestra en consola las tablas.
- Abre una ventana con la imagen en grid 4×3 (4 grupos por fila).

## 🌐 Paso 3: Ejecutar la aplicación web (Flask)

Para usar la interfaz interactiva con el bracket y el predictor personalizado:

```bash
python app.py
```

El servidor Flask se iniciará en http://127.0.0.1:5000.
Abre `index.html` en tu navegador.

## 🖥️ Uso de la interfaz web

- **Predictor personalizado**: escribe dos países en los campos de la sección superior y pulsa "Predecir". Obtendrás el marcador, el ganador y si hubo penales.
- **Bracket de eliminación directa**: los dieciseisavos de final ya están precargados con equipos. Pulsa "Predecir" en cada partido; el ganador avanzará automáticamente a la siguiente ronda.
- **Final**: cuando ambos semifinalistas estén definidos, se habilitará el botón "Predecir Final". Al pulsarlo, se mostrará el campeón.

**Importante**: la interfaz se comunica con la API Flask. Asegúrate de que `app.py` esté corriendo mientras usas el HTML.
