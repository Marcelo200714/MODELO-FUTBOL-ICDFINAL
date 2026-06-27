<<<<<<< HEAD
⚽ Predictor Mundial 2026 — Guía de uso
Este repositorio contiene un sistema de predicción de resultados de la Copa Mundial de la FIFA 2026 basado en RandomForest, con interfaz web y generación automática de la fase de grupos.

📋 Requisitos previos
Python 3.9 o superior instalado.

Gestor de paquetes pip disponible.

Opcional: entorno virtual (recomendado).

🔧 Instalación de dependencias
Abre una terminal en la carpeta del proyecto y ejecuta:

bash
pip install -r requirements.txt
Si no tienes el archivo requirements.txt, instala manualmente:

bash
pip install pandas numpy scikit-learn matplotlib flask flask-cors
🧠 Paso 1: Entrenar el modelo (si no tienes los archivos .pkl)
El proyecto necesita los modelos entrenados (model_home.pkl, model_away.pkl) y los datos procesados (df_final.pkl, forma.pkl, player_stats.pkl).
Si no los tienes, ejecuta el script de entrenamiento:

bash
python predictor_mundial_2026_v2.py
Este script:

Carga los archivos CSV (national_matches_(1992-2026).csv, player-data-full.csv, results.csv, worldcup_matches.csv).

Realiza feature engineering (incluye ranking FIFA, forma reciente, etc.).

Entrena dos RandomForestClassifier (goles local y visitante).

Guarda los archivos .pkl necesarios.

Salida esperada: verás en consola la precisión del modelo y el mensaje "✅ Modelos y datos exportados correctamente."

🖼️ Paso 2: Generar la fase de grupos (imagen 4×3)
Una vez entrenado el modelo, puedes generar las tablas de la fase de grupos con los puntajes predichos:

bash
python generar_fase_grupos.py
Esto:

Lee los partidos de la fase de grupos del Mundial 2026 desde worldcup_matches.csv.

Predice cada partido usando el modelo (con corrección exponencial por ranking).

Calcula puntos, diferencia de goles y goles a favor.

Muestra en consola las tablas.

Abre una ventana con la imagen en grid 4×3 (4 grupos por fila).

Guarda la imagen como grupos_2026.png en la carpeta actual.

Nota: Si quieres que la imagen se guarde sin abrir la ventana, comenta la línea plt.show() en el script.

🌐 Paso 3: Ejecutar la aplicación web (Flask)
Para usar la interfaz interactiva con el bracket y el predictor personalizado:

bash
python app.py
El servidor Flask se iniciará en http://127.0.0.1:5000.
Abre tu navegador y ve a esa dirección.

🖥️ Uso de la interfaz web
Predictor personalizado: escribe dos países en los campos de la sección superior y pulsa "Predecir". Obtendrás el marcador, el ganador y si hubo penales.

Bracket de eliminación directa: los dieciseisavos de final ya están precargados con equipos. Pulsa "Predecir" en cada partido; el ganador avanzará automáticamente a la siguiente ronda.

Final: cuando ambos semifinalistas estén definidos, se habilitará el botón "Predecir Final". Al pulsarlo, se mostrará el campeón.

Importante: La interfaz se comunica con la API Flask. Asegúrate de que app.py esté corriendo mientras usas el HTML.
=======
# MODELO-FUTBOL-ICD

Este proyecto predice resultados de fútbol utilizando modelos de Machine Learning.

## Cómo ejecutar localmente:
1. Clonar el repositorio.
2. Instalar las dependencias: `pip install -r requirements.txt`
3. Ejecutar la aplicación: `python app.py`
>>>>>>> 42ec1d46b111427ac0c2b2770c0c3cc9c624d31d
