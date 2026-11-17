# Proyecto Final – Hiper-heurística de selección para TSP

Dado una instancia de TSP, un clasificador predice cuál de tres algoritmos (los de la Tarea 2)
obtendrá el mejor tour: **2-opt hill climbing**, **Recocido Simulado + 2-opt** o **MST + 2-opt**.

Reporte completo: [`reporte/Reporte_Proyecto_Final.docx`](reporte/Reporte_Proyecto_Final.docx)

## Estructura

```
algoritmos_tarea2.py     # los 3 algoritmos de Tarea 2 (ejercicio3, 4 y 5)
generar_etiquetas.py     # Parte 1: corre los algoritmos sobre las 500 instancias y etiqueta
entrenar_evaluar.py      # Parte 2: Árbol / KNN, accuracy, matriz de confusión, regret, importancias
run_all.sh               # reproduce todo
data/
  instances/inst_NNNN.csv        # coordenadas (x, y) provistas
  train.csv, test.csv            # características provistas (sin costos ni etiqueta)
  {train,test}_labeled_literal.csv   # con costos y best_algorithm (parámetros de Tarea 2)
  {train,test}_labeled_t0_10.csv     # sensibilidad: SA con T0 = 10
results/                 # métricas, matrices de confusión, importancias, árbol
reporte/                 # reporte en Word
```

## Cómo correrlo

```bash
pip install -r requirements.txt
./run_all.sh          # o ejecutar los dos scripts por separado (ver run_all.sh)
```

Todo es determinista (semilla = número de instancia en los algoritmos, `random_state=42` en los modelos),
así que se reproducen exactamente las etiquetas y métricas del reporte. Las 500 instancias
tardan menos de un minuto en un solo núcleo.

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
