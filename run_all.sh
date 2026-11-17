#!/usr/bin/env bash
# Reproduce todo: etiquetas (Parte 1) + modelo y evaluación (Parte 2).
set -e
python3 generar_etiquetas.py --t0 1000 --tag literal      # etiquetas con los parámetros de Tarea 2
python3 generar_etiquetas.py --t0 10   --tag t0_10        # sensibilidad: T0 escalado a [0,1]^2
python3 entrenar_evaluar.py literal | tee results/resultados_literal.txt
