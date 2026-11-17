"""Los tres algoritmos de la Tarea 2 (ejercicio3/4/5), adaptados para correr sobre
las instancias del proyecto. La lógica es la de tus scripts; los únicos cambios son:

  * two_opt: el costo del movimiento se calcula con el delta de 4 aristas (O(1))
    en vez de recalcular tour_length completo (O(n)) dos veces por candidato.
    Es el mismo criterio (acepta si el nuevo tour es más corto) y recorre los
    vecinos en el mismo orden, solo que mucho más rápido (n=100 pasa de minutos
    a segundos). Se exige una mejora > 1e-12 para evitar ciclos por ruido flotante.
  * simulated_annealing: recibe `rng` (reproducibilidad) y devuelve el mejor tour.
    Mantiene tour_length completo en cada iteración, igual que ejercicio4.
"""
import math
import random


def tour_length(points, route):
    return sum(math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
               for i in range(len(route)))


# ----------------------------- ejercicio3 ---------------------------------
def nearest_neighbour(points, start):
    unvisited = set(range(len(points)))
    unvisited.remove(start)
    route = [start]
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda c: math.dist(points[current], points[c]))
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return route


def two_opt(points, route):
    n = len(route)
    best_route = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n):
                if k - i == 1:
                    continue
                a, b = best_route[i - 1], best_route[i]
                c, d = best_route[k], best_route[(k + 1) % n]
                delta = (math.dist(points[a], points[c]) + math.dist(points[b], points[d])
                         - math.dist(points[a], points[b]) - math.dist(points[c], points[d]))
                if delta < -1e-12:
                    best_route[i:k + 1] = best_route[i:k + 1][::-1]
                    improved = True
    return best_route


def hill_climbing_2opt(points, rng):
    """Búsqueda local: vecino más cercano (ciudad inicial aleatoria) + 2-opt."""
    start = rng.randrange(len(points))
    return two_opt(points, nearest_neighbour(points, start))


# ----------------------------- ejercicio4 ---------------------------------
def simulated_annealing(points, rng, initial_temp=1000, iterations=5000,
                        alpha=0.995, min_temp=0.01):
    """SA + movimiento 2-opt aleatorio, enfriamiento exponencial (ejercicio4)."""
    n = len(points)
    current = list(range(n))
    rng.shuffle(current)
    current_cost = tour_length(points, current)
    best, best_cost = current, current_cost
    for t in range(1, iterations + 1):
        temp = initial_temp * (alpha ** t)
        if temp <= min_temp:
            break
        i, k = sorted(rng.sample(range(n), 2))
        nxt = current[:i] + current[i:k + 1][::-1] + current[k + 1:]
        nxt_cost = tour_length(points, nxt)
        delta_e = current_cost - nxt_cost
        if delta_e > 0 or rng.random() < math.exp(delta_e / temp):
            current, current_cost = nxt, nxt_cost
            if current_cost < best_cost:
                best, best_cost = current, current_cost
    return best


# ----------------------------- ejercicio5 ---------------------------------
def mst_prim(points):
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children, root=0):
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def mst_2opt(points, rng=None):
    return two_opt(points, preorder(mst_prim(points)))
