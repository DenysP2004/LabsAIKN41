import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
# Визначення вхідних змінних
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
# Визначення вихідної змінної
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')
# Температура
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [10, 20, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [20, 40, 40])
# Вологість
humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])
# Швидкість вентилятора
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])
rule1 = ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['high'])
rule2 = ctrl.Rule(temperature['low'] | humidity['low'], fan_speed['low'])
fan_control = ctrl.ControlSystem([rule1, rule2])
fan_simulation = ctrl.ControlSystemSimulation(fan_control)
# Задаємо значення температури і вологості
fan_simulation.input['temperature'] = 30
fan_simulation.input['humidity'] = 80
# Обчислюємо вихід
fan_simulation.compute()
# Виводимо результат
print(f"Швидкість вентилятора: {fan_simulation.output['fan_speed']}")
temperature.view()
humidity.view()
fan_speed.view(sim=fan_simulation)
plt.show()