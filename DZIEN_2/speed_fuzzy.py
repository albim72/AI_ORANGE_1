import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =========================
# 1) ZMIENNE (wejścia/wyjście)
# =========================
visibility = ctrl.Antecedent(np.arange(0, 1001, 1), 'visibility')   # 0..1000 m
road_grip  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'road_grip') # 0..1
traffic    = ctrl.Antecedent(np.arange(0, 101, 1), 'traffic')       # 0..100

speed = ctrl.Consequent(np.arange(0, 151, 1), 'speed')              # 0..150 km/h

# =========================
# 2) FUNKCJE PRZYNALEŻNOŚCI
# =========================
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

# =========================
# 3) REGUŁY (IF–THEN)
# =========================
rule1 = ctrl.Rule(visibility['poor'] | road_grip['low'], speed['very_slow'])

rule2 = ctrl.Rule(visibility['medium'] & road_grip['low'], speed['slow'])
rule3 = ctrl.Rule(visibility['poor'] & traffic['high'], speed['very_slow'])

rule4 = ctrl.Rule(visibility['good'] & road_grip['high'] & traffic['low'], speed['fast'])

rule5 = ctrl.Rule(visibility['good'] & road_grip['medium'] & traffic['low'], speed['normal'])
rule6 = ctrl.Rule(traffic['high'], speed['slow'])
rule7 = ctrl.Rule(traffic['medium'] & road_grip['medium'], speed['normal'])

# =========================
# 4) SYSTEM I SYMULACJA
# =========================
car_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
car_sim = ctrl.ControlSystemSimulation(car_ctrl)

# =========================
# 5) PRZYKŁADOWE WEJŚCIA
# =========================
car_sim.input['visibility'] = 350     # m
car_sim.input['road_grip']  = 0.42    # 0..1 (np. mokro/ślisko)
car_sim.input['traffic']    = 65      # 0..100

car_sim.compute()

print("Docelowa prędkość:", round(car_sim.output['speed'], 1), "km/h")

# Opcjonalnie: wizualizacja
# visibility.view()
# road_grip.view()
# traffic.view()
# speed.view(sim=car_sim)
