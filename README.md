# KI-OPR

## Basic úloha
```python
from pulp import *
x = LpVariable("x", lowBound=0, upBound=2, cat="Integer")
y = LpVariable("y", lowBound=0, upBound=2, cat="Integer")
z = LpVariable("z", lowBound=0, upBound=2, cat="Integer")
problem = LpProblem("myProblem", LpMaximize)
problem += 3 * x + 2 * y - z
problem += x + y + z <= 1
print(problem)
status = problem.solve()
print(f"Status:\n{LpStatus[status]}\n")
print("Solution:")
for v in problem.variables():
  print(f"\t\t{v.name} = {v.varValue}")
print("\n")
print(f"Optimal solution: {problem.objective.value()}")
```

## Úloha - jídelníček
```python
from pulp.constants import LpInteger
from pulp import *

x_1 = LpVariable("Mléko", lowBound = 0, upBound=10)
x_2 = LpVariable("Jablko", lowBound = 0, upBound=10)
x_3 = LpVariable("Rýže", lowBound = 0, upBound=10)
x_4 = LpVariable("Banán", lowBound = 0, upBound=10)
x_5 = LpVariable("Chleba", lowBound = 0, upBound=10)
x_6 = LpVariable("Máslo", lowBound = 0, upBound=10)
x_7 = LpVariable("Gouda", lowBound = 0, upBound=10)
x_8 = LpVariable("Kuřecí", lowBound = 0, upBound=10)
x_9 = LpVariable("Vejce", lowBound = 0, upBound=10)
x_10 = LpVariable("Pivo", lowBound = 0, upBound=10)
x_11 = LpVariable("Buřt", lowBound = 0, upBound=10)

model = LpProblem(name = "Výživa", sense = LpMinimize)
model += 3 * x_1 + 5 * x_2 + 4 * x_3 + 4 * x_4 + 4.4 * x_5 + 24 * x_6 + 17 * x_7 + 12 * x_8 + 10 * x_9 + 8* x_10 + 10 * x_11

model +=  260 * x_1 + 218 * x_2 + 1448 * x_3 + 394 * x_4 + 244 * x_5 + 3134 * x_6 + 1337 * x_7 + 442 * x_8 + 632 * x_9 + 175 * x_10 + 1313 * x_11   >= 8000
model +=  3 * x_1 + 0 * x_2 + 3 * x_3 + 1.2 * x_4 + 8 * x_5 + 27 * x_6 + 25 * x_7 + 23 * x_8 + 13 * x_9 + 0.5 * x_10 + 10 * x_11                    >= 72
model +=  4 * x_1 + 17 * x_2 + 28 * x_3 + 25 * x_4 + 45 * x_5 + 0.8 * x_6 + 0.5 * x_7 + 0 * x_8 + 1.1 * x_9 + 3 * x_10 + 2.8 * x_11                 >= 200
model +=  3.5 * x_1 + 0 * x_2 + 0.2 * x_3 + 30 * x_4 + 3.2 * x_5 + 18 * x_6 + 27 * x_7 + 31 * x_8 + 10 * x_9 + 0.22 * x_10 + 30 * x_11              >= 50

model +=  260 * x_1 + 218 * x_2 + 1448 * x_3 + 394 * x_4 + 244 * x_5 + 3134 * x_6 + 1337 * x_7 + 442 * x_8 + 632 * x_9 + 175 * x_10 + 1313 * x_11 <= 12000
model +=  3 * x_1 + 0 * x_2 + 3 * x_3 + 1.2 * x_4 + 8 * x_5 + 27 * x_6 + 25 * x_7 + 23 * x_8 + 13 * x_9 + 0.5 * x_10 + 10 * x_11 <= 140
model +=  4 * x_1 + 17 * x_2 + 28 * x_3 + 25 * x_4 + 45 * x_5 + 0.8 * x_6 + 0.5 * x_7 + 0 * x_8 + 1.1 * x_9 + 3 * x_10 + 2.8 * x_11 <= 400
model +=  3.5 * x_1 + 0 * x_2 + 0.2 * x_3 + 30 * x_4 + 3.2 * x_5 + 18 * x_6 + 27 * x_7 + 31 * x_8 + 10 * x_9 + 0.22 * x_10 + 30 * x_11 <= 80

print(model)
status = model.solve()
print(f"Status:\n{LpStatus[status]}\n")
print("Optimální řešení:")
for v in model.variables():
  print(f"\t\t{v.name} = {v.varValue} x 100g")
print("\n")
print(f"Celková cena: {model.objective.value()} Kč")
```
## Úloha - proložení bodů přímkou
```python
from pulp import *
import matplotlib.pyplot as plt

# beta 0, beta 1 ... směrnice přímky
# x, y ... souřadnice
# epsilon ... chyba, je jich tolik, kolik je bodů

x = LpVariable("x")
y = LpVariable("y")
# z = LpVariable("z")
epsilon = LpVariable("epsilon", lowBound = 0)

model = LpProblem(name = "Přímka", sense = LpMinimize)
body = [[0,-1.5],[1,2.2],[2,10],[3,19.5],[4,33],[5,56]]

for i, (b_1,b_2) in enumerate(body):
  odchylka = LpVariable(f"odchylka {i}", lowBound = 0)
  model += x * b_1 + y - b_2 <= odchylka
  model += -(x * b_1 + y - b_2) <= odchylka
  model += odchylka <= epsilon

model += epsilon
model.solve()

x_výsledek = x.varValue
y_výsledek = y.varValue
epsilon_výsledek = epsilon.varValue

print(f"Rovnice přímky: f(x) = {x_výsledek}x + {y_výsledek}")
print(f"Maximální odchylka (epsilon): {epsilon_výsledek}")


x_souradnice = [bod[0] for bod in body] 
y_souradnice = [bod[1] for bod in body]  
plt.figure(figsize = (8,6))
plt.scatter(x_souradnice, y_souradnice, color="deeppink", label="Data")
x_primka = range(6)  
y_primka = [x_výsledek * x + y_výsledek for x in x_primka] 
plt.plot(x_primka, y_primka, color="navy", label="Přímka")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Přímka s nejmenší odchylkou")
plt.legend()
plt.grid(True)
plt.show()
```

