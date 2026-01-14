import math
import random

# ------- Dane: kilka punktów w 2D (możesz zmienić / dodać) -------
CITIES = [
    (0, 0),
    (1, 5),
    (5, 2),
    (6, 6),
    (2, 3),
    (7, 3),
    (3, 7),
    (4, 4),
    (8, 1),
    (2, 8),
]

random.seed(42)  # dla powtarzalności

# ------- Pomocnicze funkcje -------
def dist(a, b):
    """Odległość Euklidesowa między punktami a i b."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tour_length(tour):
    """Długość trasy (domykamy pętlę do miasta startowego)."""
    total = 0.0
    for i in range(len(tour)):
        a = CITIES[tour[i]]
        b = CITIES[tour[(i + 1) % len(tour)]]
        total += dist(a, b)
    return total

def random_neighbor(tour):
    """Prosty sąsiad: zamiana (swap) dwóch losowych miast."""
    i, j = random.sample(range(len(tour)), 2)
    new_tour = tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# ------- Simulated Annealing -------
def simulated_annealing(
    initial_tour,
    T_start=10.0,      # temperatura początkowa
    T_end=1e-3,        # temperatura końcowa
    alpha=0.995,       # współczynnik chłodzenia (0<alpha<1)
    iters_per_T=200,   # ile prób na poziom temperatury
):
    current = initial_tour[:]
    current_cost = tour_length(current)
    best = current[:]
    best_cost = current_cost

    T = T_start
    steps = 0

    while T > T_end:
        for _ in range(iters_per_T):
            steps += 1
            candidate = random_neighbor(current)
            cand_cost = tour_length(candidate)
            delta = cand_cost - current_cost

            if delta <= 0:
                # lepsze -> akceptuj
                current, current_cost = candidate, cand_cost
                if cand_cost < best_cost:
                    best, best_cost = candidate[:], cand_cost
            else:
                # gorsze -> akceptuj z prawdopodobieństwem exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    current, current_cost = candidate, cand_cost

        # chłodzenie
        T *= alpha

    return best, best_cost, steps

# ------- Uruchomienie -------
n = len(CITIES)
initial = list(range(n))
random.shuffle(initial)

best_tour, best_len, steps = simulated_annealing(
    initial_tour=initial,
    T_start=10.0,
    T_end=1e-3,
    alpha=0.995,
    iters_per_T=200
)

print("Miasta (indeks -> współrzędne):")
for i, p in enumerate(CITIES):
    print(f"{i}: {p}")

print("\nTrasa startowa:", initial, " | długość =", round(tour_length(initial), 3))
print("Najlepsza trasa:", best_tour, " | długość =", round(best_len, 3))
print("Liczba kroków:", steps)
