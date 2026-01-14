import random
from typing import Dict, Any, List, Tuple

# ==========
#  GA CORE
# ==========

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _uniform(a: float, b: float) -> float:
    return random.uniform(a, b)

def _gaussian(mu: float, sigma: float) -> float:
    return random.gauss(mu, sigma)

class SimpleGA:
    """
    Minimalny GA:
      - genom: dict {gen: wartość}
      - selekcja: turniejowa
      - krzyżowanie: jednolite (50/50 per gen)
      - mutacja: gaussowska z klipowaniem
    """
    def __init__(self, config: Dict[str, Any], fitness_fn):
        self.genes_spec: Dict[str, Tuple[float, float]] = config["genes"]
        self.pop_size: int = int(config.get("population", 40))
        self.generations: int = int(config.get("generations", 60))
        self.mutation_rate: float = float(config.get("mutation_rate", 0.1))
        self.mutation_sigma: float = float(config.get("mutation_sigma", 0.1))
        self.tournament_k: int = int(config.get("tournament_k", 3))
        seed = config.get("seed", None)
        if seed is not None:
            random.seed(seed)
        self.fitness_fn = fitness_fn

    def random_genome(self) -> Dict[str, float]:
        return {g: _uniform(lo, hi) for g, (lo, hi) in self.genes_spec.items()}

    def crossover(self, a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        return {g: (a[g] if random.random() < 0.5 else b[g]) for g in self.genes_spec}

    def mutate(self, genome: Dict[str, float]) -> Dict[str, float]:
        out = genome.copy()
        for g, (lo, hi) in self.genes_spec.items():
            if random.random() < self.mutation_rate:
                out[g] = _clip(_gaussian(out[g], self.mutation_sigma), lo, hi)
        return out

    def tournament_select(self, pop, fits):
        idxs = random.sample(range(len(pop)), k=self.tournament_k)
        best_idx = max(idxs, key=lambda i: fits[i])
        return pop[best_idx]

    def run(self):
        population = [self.random_genome() for _ in range(self.pop_size)]
        fitness = [self.fitness_fn(g) for g in population]

        best_idx = max(range(self.pop_size), key=lambda i: fitness[i])
        best_genome = population[best_idx]
        best_fit = fitness[best_idx]

        for _ in range(self.generations):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(population, fitness)
                p2 = self.tournament_select(population, fitness)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)

            population = new_pop
            fitness = [self.fitness_fn(g) for g in population]

            gen_best_idx = max(range(self.pop_size), key=lambda i: fitness[i])
            if fitness[gen_best_idx] > best_fit:
                best_fit = fitness[gen_best_idx]
                best_genome = population[gen_best_idx]

        return best_genome, best_fit


# ==========================
#  METAKLASA Z NADZOREM GA
# ==========================

class GAOptimizingMeta(type):
    """
    Metaklasa:
      1) po utworzeniu klasy odczytuje __ga_config__,
      2) pobiera callable fitness z cls (obsługuje @classmethod/@staticmethod),
      3) uruchamia GA i wstrzykuje najlepszy genom w atrybuty klasy,
      4) zapisuje __best_genome__ oraz __best_fitness__.
    """
    def __new__(mcls, name, bases, namespace, **kwargs):
        ga_cfg = namespace.get("__ga_config__", None)

        # Utwórz klasę najpierw:
        cls = super().__new__(mcls, name, bases, dict(namespace), **kwargs)

        if ga_cfg is not None:
            fitness_callable = getattr(cls, "fitness", None)
            if callable(fitness_callable):
                ga = SimpleGA(ga_cfg, fitness_callable)
                best_genome, best_fit = ga.run()

                for k, v in best_genome.items():
                    setattr(cls, k, v)
                setattr(cls, "__best_genome__", best_genome)
                setattr(cls, "__best_fitness__", best_fit)
            else:
                # Brak fitness() lub nie-callable → pomiń GA (lub rzuć wyjątek wedle potrzeb)
                pass

        return cls


# ====================================
#  PRZYKŁAD: Klasa z GA w metaklasie
# ====================================

class ThresholdClassifier(metaclass=GAOptimizingMeta):
    """
    Prosty klasyfikator 1D:
      pred = 1 jeśli (weight * x) >= threshold, w przeciwnym razie 0.
    Metaklasa dobiera (threshold, weight) maksymalizując trafność na TRAIN.
    """

    TRAIN: List[Tuple[float, int]] = [
        (-1.0, 0), (-0.5, 0), (0.0, 0), (0.2, 0),
        (0.8, 1), (1.0, 1), (1.5, 1), (2.0, 1)
    ]

    __ga_config__ = {
        "genes": {
            "threshold": (-1.0, 2.0),
            "weight":    (-3.0, 3.0),
        },
        "population": 40,
        "generations": 60,
        "mutation_rate": 0.15,
        "mutation_sigma": 0.15,
        "tournament_k": 3,
        "seed": 7,
    }

    @classmethod
    def fitness(cls, genome: Dict[str, float]) -> float:
        thr = genome["threshold"]
        w   = genome["weight"]
        correct = 0
        for x, y in cls.TRAIN:
            pred = 1 if (w * x) >= thr else 0
            if pred == y:
                correct += 1
        return correct / len(cls.TRAIN)

    def predict(self, x: float) -> int:
        return 1 if (self.weight * x) >= self.threshold else 0


# ==========
#  DEMO
# ==========

if __name__ == "__main__":
    print("Najlepszy genom (wstrzyknięty do klasy):", ThresholdClassifier.__best_genome__)
    print("Najlepszy fitness:", round(ThresholdClassifier.__best_fitness__, 3))
    print("Atrybuty klasy → threshold, weight:",
          ThresholdClassifier.threshold, ThresholdClassifier.weight)

    clf = ThresholdClassifier()
    test_points = [-1.0, -0.2, 0.1, 0.9, 1.7]
    print("\nPredykcje:")
    for x in test_points:
        print(f"x={x:>4}: pred={clf.predict(x)}")

    print("____________________________________________")
    class MojaNowa(ThresholdClassifier):
      def fitness(self, genome: Dict[str, float]) -> float:pass

    mn = MojaNowa()
    test_points = [-1.1, -0.03, 0.3, 1,4, 2.3]
    print("\nPredykcje:")
    for x in test_points:
        print(f"x={x:>4}: pred={clf.predict(x)}")
