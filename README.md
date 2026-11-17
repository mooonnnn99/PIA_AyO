# Proyecto Final – Hiper-heurística de selección para TSP

Dada una instancia de TSP, un clasificador predice cuál de tres algoritmos (los de la Tarea 2)
obtendrá el mejor tour: **2-opt hill climbing**, **Recocido Simulado + 2-opt** o **MST + 2-opt**.

Reporte completo: [`reporte/Reporte_Proyecto_Final.docx`](reporte/Reporte_Proyecto_Final.docx)

## Estructura

```
algoritmos_tarea2.py     # los 3 algoritmos de Tarea 2 (ejercicio3, 4 y 5)
generar_etiquetas.py     # Parte 1: corre los algoritmos sobre las 500 instancias y etiqueta
entrenar_evaluar.py      # Parte 2: Árbol / KNN, accuracy, matriz de confusión, regret, importancias
run_all.sh               # reproduce todo
requirements.txt         # dependencias
data.zip                 # carpeta data/ comprimida (ver "Cómo correrlo"); al descomprimir contiene:
                         #   data/instances/inst_NNNN.csv         coordenadas (x, y) provistas
                         #   data/train.csv, data/test.csv        características provistas (sin costos ni etiqueta)
                         #   data/{train,test}_labeled_literal.csv  con costos y best_algorithm (parámetros de Tarea 2)
                         #   data/{train,test}_labeled_t0_10.csv    sensibilidad: SA con T0 = 10
results/                 # métricas, matrices de confusión, importancias, árbol
reporte/                 # reporte en Word
```

## Cómo correrlo

**1. Descomprimir los datos.** En la raíz del repo, junto a los scripts:

```bash
unzip data.zip           # en Windows: clic derecho > Extraer aquí
```

Debe quedar una carpeta `data/` con `instances/inst_0001.csv` ... `inst_0500.csv`, `train.csv` y `test.csv`.
Si aparece `data/data/`, mueve el contenido un nivel arriba: los scripts leen de `data/` en la raíz del repo.

**2. Instalar dependencias y correr:**

```bash
pip install -r requirements.txt
bash run_all.sh
```

En Windows, si `python3` no existe, reemplázalo por `python` en `run_all.sh`, o ejecuta a mano:

```bash
python generar_etiquetas.py --t0 1000 --tag literal
python generar_etiquetas.py --t0 10 --tag t0_10
python entrenar_evaluar.py literal
```

Probado con Python 3.12, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4 y matplotlib 3.11.2.
Todo es determinista (semilla = número de instancia en los algoritmos, `random_state=42` en los modelos),
así que se reproducen las etiquetas y métricas del reporte con estas versiones; en otras
versiones podrían cambiar algunos decimales de los modelos. Las 500 instancias tardan
menos de un minuto en un solo núcleo.

## Parámetros del etiquetado

| Algoritmo | Configuración |
|---|---|
| 2-opt hill climbing | Vecino más cercano (ciudad inicial aleatoria con semilla) + 2-opt hasta óptimo local |
| SA + 2-opt | Tour inicial aleatorio, movimiento 2-opt aleatorio, T0 = 1000, T = T0·0.995^t, máx. 5000 iteraciones (se corta a T ≤ 0.01, iteración 2297), se devuelve el mejor tour visitado |
| MST + 2-opt | Prim + recorrido en preorden + 2-opt |

- El costo de cada movimiento 2-opt se calcula con el delta de 4 aristas (equivalente a recalcular el tour completo, pero O(1)).
- Empates (costos iguales a 1e-9): gana el primero en el orden 2opt_hill_climbing, sa_2opt, mst_2opt.

## Resultados (test, 149 instancias)

| Modelo | Accuracy | Regret relativo medio |
|---|---|---|
| Baseline: siempre 2-opt (58.8% de las 500 instancias) | 52.3% | 1.56% |
| Árbol de decisión | 51.7% | 1.66% |
| KNN (k=5) | 53.0% | 1.78% |
| Árbol balanceado | 48.3% | 2.34% |

Ningún modelo supera de forma significativa al baseline; ver el reporte para el análisis.
