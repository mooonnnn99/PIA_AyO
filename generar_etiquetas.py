"""Parte 1: corre los 3 algoritmos de Tarea 2 sobre cada instancia y etiqueta.

Uso: python3 generar_etiquetas.py [--t0 1000] [--tag literal]
Escribe data/train_labeled_<tag>.csv y data/test_labeled_<tag>.csv
"""
import argparse, csv, random, sys
import pandas as pd
from algoritmos_tarea2 import (tour_length, hill_climbing_2opt, mst_2opt,
                               simulated_annealing)

TOL = 1e-9   # costos a menos de TOL se consideran empate
ORDER = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]  # prioridad en empates


def load_points(inst_id):
    with open(f"data/instances/{inst_id}.csv") as f:
        r = csv.reader(f); next(r)
        return [(float(x), float(y)) for x, y in r]


def label(costs):
    m = min(costs.values())
    winners = [a for a in ORDER if costs[a] <= m + TOL]
    return winners[0], len(winners) > 1


def run_instance(inst_id, t0):
    seed = int(inst_id.split("_")[1])
    pts = load_points(inst_id)
    costs = {
        "2opt_hill_climbing": tour_length(pts, hill_climbing_2opt(pts, random.Random(seed))),
        "sa_2opt": tour_length(pts, simulated_annealing(pts, random.Random(seed), initial_temp=t0)),
        "mst_2opt": tour_length(pts, mst_2opt(pts)),
    }
    best, tie = label(costs)
    return costs, best, tie


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=1000)
    ap.add_argument("--tag", default="literal")
    a = ap.parse_args()
    for split in ["train", "test"]:
        df = pd.read_csv(f"data/{split}.csv")
        out = []
        for k, inst in enumerate(df.instance_id):
            c, best, tie = run_instance(inst, a.t0)
            out.append({"cost_2opt_hill_climbing": c["2opt_hill_climbing"],
                        "cost_sa_2opt": c["sa_2opt"], "cost_mst_2opt": c["mst_2opt"],
                        "best_algorithm": best, "tie": tie})
            if k % 50 == 0:
                print(split, k, "/", len(df), file=sys.stderr, flush=True)
        df = pd.concat([df, pd.DataFrame(out)], axis=1)
        df.to_csv(f"data/{split}_labeled_{a.tag}.csv", index=False)
        print(split, "listo")
