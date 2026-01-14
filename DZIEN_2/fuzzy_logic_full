import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =========================================================
# 1) ZMIENNE (wejścia/wyjście)
# =========================================================
visibility = ctrl.Antecedent(np.arange(0, 1001, 1), 'visibility')    # 0..1000 m
road_grip  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'road_grip')  # 0..1
traffic    = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic')        # 0..100

speed = ctrl.Consequent(np.arange(0, 151, 1), 'speed')               # 0..150 km/h

# =========================================================
# 2) FUNKCJE PRZYNALEŻNOŚCI
# =========================================================
# Widoczność
visibility['poor']   = fuzz.trapmf(visibility.universe, [0, 0, 120, 250])
visibility['medium'] = fuzz.trimf(visibility.universe, [200, 450, 700])
visibility['good']   = fuzz.trapmf(visibility.universe, [650, 800, 1000, 1000])

# Przyczepność
road_grip['low']    = fuzz.trapmf(road_grip.universe, [0.0, 0.0, 0.25, 0.45])
road_grip['medium'] = fuzz.trimf(road_grip.universe, [0.35, 0.55, 0.75])
road_grip['high']   = fuzz.trapmf(road_grip.universe, [0.65, 0.80, 1.0, 1.0])

# Ruch
traffic['low']    = fuzz.trapmf(traffic.universe, [0, 0, 20, 40])
traffic['medium'] = fuzz.trimf(traffic.universe, [30, 55, 80])
traffic['high']   = fuzz.trapmf(traffic.universe, [70, 85, 100, 100])

# Prędkość (wyjście)
speed['very_slow'] = fuzz.trimf(speed.universe, [0, 0, 35])
speed['slow']      = fuzz.trimf(speed.universe, [20, 45, 70])
speed['normal']    = fuzz.trimf(speed.universe, [60, 85, 110])
speed['fast']      = fuzz.trimf(speed.universe, [95, 120, 150])

# =========================================================
# 3) GENERATOR "WSZYSTKICH MOŻLIWYCH" REGUŁ (27)
# =========================================================
VIS_LABELS = ['poor', 'medium', 'good']
GRIP_LABELS = ['low', 'medium', 'high']
TRAFFIC_LABELS = ['low', 'medium', 'high']

# Prosta „polityka” mapowania kombinacji -> kategoria prędkości.
# Im gorzej, tym bardziej w lewo (very_slow).
# Możesz stroić progi i wagi: to jest świadoma, czytelna heurystyka.
SCORE_VIS = {'poor': -2, 'medium': -1, 'good': 0}
SCORE_GRIP = {'low': -2, 'medium': -1, 'high': 0}
SCORE_TRAFFIC = {'low': 0, 'medium': -1, 'high': -2}

def consequent_from_labels(v_label: str, g_label: str, t_label: str) -> str:
    score = SCORE_VIS[v_label] + SCORE_GRIP[g_label] + SCORE_TRAFFIC[t_label]
    # najlepszy przypadek: 0 -> fast
    # lekko pogorszone: -1, -2 -> normal
    # gorzej: -3, -4 -> slow
    # bardzo źle: <= -5 -> very_slow
    if score == 0:
        return 'fast'
    elif score >= -2:
        return 'normal'
    elif score >= -4:
        return 'slow'
    else:
        return 'very_slow'

rules = []
rule_meta = []  # do raportowania: opis reguły + jej docelowa etykieta prędkości

for v in VIS_LABELS:
    for g in GRIP_LABELS:
        for t in TRAFFIC_LABELS:
            out_label = consequent_from_labels(v, g, t)
            r = ctrl.Rule(visibility[v] & road_grip[g] & traffic[t], speed[out_label])
            rules.append(r)

            rule_meta.append({
                "vis": v, "grip": g, "traffic": t,
                "out": out_label,
                "text": f"IF visibility is {v} AND road_grip is {g} AND traffic is {t} THEN speed is {out_label}"
            })

car_ctrl = ctrl.ControlSystem(rules)
car_sim = ctrl.ControlSystemSimulation(car_ctrl)

# =========================================================
# 4) NARZĘDZIA DO RAPORTOWANIA: KTÓRE REGUŁY "ZAPALIŁY"
# =========================================================
def membership_degrees(x_visibility: float, x_grip: float, x_traffic: float):
    """Zwraca stopnie przynależności (mu) dla wszystkich etykiet."""
    mu_vis = {
        lab: fuzz.interp_membership(visibility.universe, visibility[lab].mf, x_visibility)
        for lab in VIS_LABELS
    }
    mu_grip = {
        lab: fuzz.interp_membership(road_grip.universe, road_grip[lab].mf, x_grip)
        for lab in GRIP_LABELS
    }
    mu_traffic = {
        lab: fuzz.interp_membership(traffic.universe, traffic[lab].mf, x_traffic)
        for lab in TRAFFIC_LABELS
    }
    return mu_vis, mu_grip, mu_traffic

def firing_strength_for_rule(mu_vis, mu_grip, mu_traffic, v_label, g_label, t_label):
    """
    Dla reguł typu AND/AND/AND w Mamdanim:
    firing = min( mu(v), mu(g), mu(t) )
    """
    return min(mu_vis[v_label], mu_grip[g_label], mu_traffic[t_label])

def explain_rules(x_visibility: float, x_grip: float, x_traffic: float, top_k: int = 7):
    """Wypisuje TOP reguł wg firing strength oraz regułę dominującą."""
    mu_vis, mu_grip, mu_traffic = membership_degrees(x_visibility, x_grip, x_traffic)

    fired = []
    for meta in rule_meta:
        fs = firing_strength_for_rule(mu_vis, mu_grip, mu_traffic, meta["vis"], meta["grip"], meta["traffic"])
        fired.append((fs, meta))

    fired.sort(key=lambda z: z[0], reverse=True)

    dominant_fs, dominant_meta = fired[0]
    print("\n=== Wyjaśnienie reguł (firing strength) ===")
    print(f"Reguła dominująca: {dominant_meta['text']}")
    print(f"Siła odpalenia (dominant): {dominant_fs:.4f}")

    print(f"\nTOP {top_k} reguł:")
    for i, (fs, meta) in enumerate(fired[:top_k], start=1):
        print(f"{i:>2}. fs={fs:.4f} | {meta['text']}")

# =========================================================
# 5) FUNKCJA URUCHOMIENIA SYMULACJI
# =========================================================
def run_case(x_visibility: float, x_grip: float, x_traffic: float, show_top_rules: int = 7):
    car_sim.input['visibility'] = float(x_visibility)
    car_sim.input['road_grip']  = float(x_grip)
    car_sim.input['traffic']    = float(x_traffic)

    car_sim.compute()
    out_speed = float(car_sim.output['speed'])

    print("\n==============================")
    print("WEJŚCIA:")
    print(f"  visibility = {x_visibility} m")
    print(f"  road_grip  = {x_grip} (0..1)")
    print(f"  traffic    = {x_traffic} (0..100)")
    print("WYJŚCIE:")
    print(f"  Docelowa prędkość = {out_speed:.1f} km/h")

    explain_rules(x_visibility, x_grip, x_traffic, top_k=show_top_rules)

    return out_speed

# =========================================================
# 6) PRZYKŁADOWE WEJŚCIA (jak u Ciebie)
# =========================================================
run_case(350, 0.42, 65, show_top_rules=7)
run_case(220, 0.30, 65, show_top_rules=7)
run_case(350, 0.85, 65, show_top_rules=7)

# Opcjonalnie: wykresy
# visibility.view()
# road_grip.view()
# traffic.view()
# speed.view(sim=car_sim)
