"""Parte 2: clasificador de best_algorithm + evaluación (accuracy, matriz de confusión, regret)."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.inspection import permutation_importance

os.makedirs("results", exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "literal"
SEED = 42
CLASSES = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
COSTCOL = {"2opt_hill_climbing": "cost_2opt_hill_climbing", "sa_2opt": "cost_sa_2opt", "mst_2opt": "cost_mst_2opt"}
NUM = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]

tr = pd.read_csv(f"data/train_labeled_{TAG}.csv"); te = pd.read_csv(f"data/test_labeled_{TAG}.csv")

def X(df):
    d = df[NUM].copy()
    for dist in ["uniform", "clustered", "border"]:
        d[f"dist_{dist}"] = (df.distribution == dist).astype(int)
    return d
Xtr, Xte, ytr, yte = X(tr), X(te), tr.best_algorithm, te.best_algorithm

# ---------- balance de clases (500 instancias) y baseline ----------
allv = pd.concat([tr, te]).best_algorithm.value_counts()
majority = allv.index[0]
print("Balance 500 inst.:\n", pd.DataFrame({"n": allv, "%": (allv/500*100).round(1)}).to_string())
print("Clase mayoritaria:", majority)

def regret(df, pred):
    chosen = np.array([r[COSTCOL[p]] for p, (_, r) in zip(pred, df.iterrows())])
    best = df[list(COSTCOL.values())].min(axis=1).values
    return chosen - best, (chosen - best) / best

def evaluate(name, pred):
    ab, rel = regret(te, pred)
    return {"modelo": name, "accuracy": accuracy_score(yte, pred),
            "balanced_acc": balanced_accuracy_score(yte, pred, ),
            "regret_abs_medio": ab.mean(), "regret_rel_medio_%": rel.mean()*100,
            "regret_cero_%": (ab < 1e-9).mean()*100}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = {
 "Arbol": GridSearchCV(DecisionTreeClassifier(random_state=SEED),
          {"max_depth": [2, 3, 4, 5, 6, None], "min_samples_leaf": [1, 3, 5, 10]}, cv=cv, scoring="accuracy"),
 "Arbol balanceado": GridSearchCV(DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
          {"max_depth": [2, 3, 4, 5, 6, None], "min_samples_leaf": [1, 3, 5, 10]}, cv=cv, scoring="balanced_accuracy"),
 "KNN": GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),
          {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 11, 15, 21],
           "kneighborsclassifier__weights": ["uniform", "distance"]}, cv=cv, scoring="accuracy"),
}
rows = [evaluate(f"Baseline: siempre {majority}", np.array([majority]*len(te))),
        evaluate("Baseline: siempre 2opt_hill_climbing", np.array(["2opt_hill_climbing"]*len(te)))]
fitted, preds = {}, {}
for name, gs in models.items():
    gs.fit(Xtr, ytr); fitted[name] = gs.best_estimator_; preds[name] = gs.predict(Xte)
    r = evaluate(name, preds[name]); r["cv_score"] = gs.best_score_; r["params"] = str(gs.best_params_); rows.append(r)
# oracle: la mejor etiqueta posible (regret 0 por construcción)
res = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print("\nTEST (149 inst.)\n", res.round(3).to_string(index=False))
res.to_csv(f"results/resultados_{TAG}.csv", index=False)

# ---------- matrices de confusión ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, name in zip(axs, models):
    cm = confusion_matrix(yte, preds[name], labels=CLASSES)
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["HC", "SA", "MST"]); ax.set_yticklabels(["HC", "SA", "MST"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(name)
    print(f"\nMatriz de confusión {name} (filas=real, cols=pred; HC, SA, MST)\n", cm)
plt.tight_layout(); plt.savefig(f"results/matrices_confusion_{TAG}.png", dpi=130); plt.close()

# ---------- importancia de características (árbol y permutación, mejor árbol) ----------
tree = fitted["Arbol"]
imp = pd.Series(tree.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
pi = permutation_importance(tree, Xte, yte, n_repeats=30, random_state=SEED)
pimp = pd.Series(pi.importances_mean, index=Xtr.columns).sort_values(ascending=False)
print("\nImportancia (Gini, árbol):\n", imp.round(3).to_string())
print("\nImportancia por permutación (test, caída de accuracy):\n", pimp.round(3).to_string())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
imp[::-1].plot.barh(ax=ax[0]); ax[0].set_title("Gini (árbol)")
pimp[::-1].plot.barh(ax=ax[1]); ax[1].set_title("Permutación (test)")
plt.tight_layout(); plt.savefig(f"results/importancias_{TAG}.png", dpi=130); plt.close()
plt.figure(figsize=(14, 7)); plot_tree(tree, feature_names=list(Xtr.columns), class_names=tree.classes_, filled=True, fontsize=7)
plt.savefig(f"results/arbol_{TAG}.png", dpi=130); plt.close()

# ---------- patrón por tamaño / distribución y desempeño del árbol ----------
full = pd.concat([tr, te], ignore_index=True)
print("\nGanador por n:\n", pd.crosstab(full.n, full.best_algorithm).to_string())
print("\nGanador por distribución:\n", pd.crosstab(full.distribution, full.best_algorithm).to_string())
print("\nGanador por n (solo victorias estrictas, sin empates):\n", pd.crosstab(full[~full.tie].n, full[~full.tie].best_algorithm).to_string())
te2 = te.assign(pred=preds["Arbol"], ok=(preds["Arbol"] == yte.values))
print("\nAccuracy del árbol por n:\n", te2.groupby("n").ok.mean().round(2).to_string())
print("\nAccuracy del árbol por distribución:\n", te2.groupby("distribution").ok.mean().round(2).to_string())
print("\nPredicciones del árbol:", pd.Series(preds["Arbol"]).value_counts().to_dict())
print("\nRegret medio por clase real (árbol):")
ab, rel = regret(te, preds["Arbol"]); print(pd.Series(rel*100, index=te.index).groupby(te.best_algorithm).mean().round(2).to_string())
