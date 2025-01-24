# KI-OPR

## Duality
- [Duality Problem 1,2 - Linear Programming Problems (LPP)](https://www.youtube.com/watch?v=drU7dOa836M)
### unrestricted variables
- [Duality Problem 3 - Linear Programming Problems (LPP)](https://www.youtube.com/watch?v=TiEdWbAVpeg)  
- [Duality Problem 4 - Linear Programming Problems (LPP)](https://www.youtube.com/watch?v=fCxrNxnnmS0)  
- [Duality Problem 5 - Linear Programming Problems (LPP)](https://www.youtube.com/watch?v=c6Ak_MOWwRo)

## Simplex method
### One phase method
- [Intro to Simplex Method | Solve LP | Simplex Tableau](https://www.youtube.com/watch?v=9YKLXFqCy6E&list=PLZTb54-HTBOPtg03d59zNSaRZuJ1hyUxc&index=2)  
### Big M method
- [Simplex Method 2 | Big M Tableau | Minimization Problem](https://www.youtube.com/watch?v=btjxqq-vMOg)  
### Two phase method
- [2 Phase Method](https://www.youtube.com/watch?v=1iQv66APoTQ)

## Transportation problem
### Phase 1
- Finding **initial basic feasible solution**: 
#### North-west method
- [Transportation Model | North-West Corner method - Balanced Model | Operations Research](https://www.youtube.com/watch?v=VIPtuo8zEhs)  
#### Least cost method - lepší pro MODI method
- [Transportation Model | Least Cost method - Balanced Model | Operations Research](https://www.youtube.com/watch?v=1EGXueT-_ig)  
### Phase 2
- Finding optimal solution:
#### MODI method (8:16)
- [Transportation problem [ MODI method - U V method with Optimal Solution ]](https://www.youtube.com/watch?v=-w2z3MVTcQA&list=PLZTb54-HTBOPtg03d59zNSaRZuJ1hyUxc&index=9)

## Assignment problem
### Hungarian method
- [How to Solve an Assignment Problem Using the Hungarian Method](https://www.youtube.com/watch?v=ezSx8OyBZVc&list=PLZTb54-HTBOPtg03d59zNSaRZuJ1hyUxc&index=10)
#### Unbalanced problem
- [Assignment model, Part-5 : Unbalanced assignment problems](https://www.youtube.com/watch?v=UrnAZJ9iy_s)
## CPM
- tady řeší uplně stejnou úlohu jakou máme zadanou na moodlu  
- [Crashing of a Project Network - Example 4 | An Important Concept | CPM | PERT | Easy Method](https://www.youtube.com/watch?v=SJTJLWFw7rE&list=PLZTb54-HTBOPtg03d59zNSaRZuJ1hyUxc&index=1)  

- **Další úlohy:**
- [Crashing Of Project Network - Example 1 | 3 Critical Paths](https://www.youtube.com/watch?v=cPg71FOUdYM)  
- [Crashing of a Project Network - Example 2 | Big Network](https://www.youtube.com/watch?v=orqJkZGpJp8&t=1876s)  
- [Crashing of a Project Network - Example 3 | Three Critical paths](https://www.youtube.com/watch?v=bLkkTHpm2Wk)  
- [Crashing of a Project Network - Example 5 | A Tricky Problem](https://www.youtube.com/watch?v=5qpcNEb6cL0)  
- [Crashing of a Project Network - Example 6 | Big Network having 2 Critical paths](https://www.youtube.com/watch?v=HZzE8alRFk0)

## PERT
- stejný úlohy jako sme údajně dělali na cviku  
- [PERT - Project Evaluation Review and Technique in Project Management](https://www.youtube.com/watch?v=WrAf6zdteXI&list=PLZTb54-HTBOPtg03d59zNSaRZuJ1hyUxc&index=11)



## Kódy 

- u zápočtu si můžem vybrat jestli chcem použít python nebo matlab 
### Basic úloha
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

### Úloha - jídelníček
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
### Úloha - proložení bodů přímkou
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

### Obchodní cestujííc the fuck
```python
import numpy as np 
from pulp import *
import random

 # počet měst
n = 6

# generování náhodné matice
time_matrix = np.zeros((n, n))  
for i in range(n):
    for j in range(i + 1, n):  
        time_matrix[i, j] = random.randint(1, 20)  
        time_matrix[j, i] = time_matrix[i, j] 
row,col = time_matrix.shape

problem = LpProblem('Problém cestujícího obchodníka', LpMinimize)

decisionVariableX = LpVariable.dicts('decisionVariable_X', ((i, j) for i in range(row) for j in range(row)), lowBound=0, upBound=1, cat='Integer')

decisionVariableU = LpVariable.dicts('decisionVariable_U', (i for i in range(row)), lowBound=1, cat='Integer')

problem += lpSum(time_matrix[i][j] * decisionVariableX[i, j] for i in range(row) for j in range(row))

for i in range(row):
  problem += (decisionVariableX[i,i] == 0) 
  problem += lpSum(decisionVariableX[i,j] for j in range(row))==1 
  problem += lpSum(decisionVariableX[j,i] for j in range(row)) ==1 
  for j in range(row):
    if i != j and (i != 0 and j != 0):
        problem += decisionVariableU[i]  <=  decisionVariableU[j] + row * (1 - decisionVariableX[i, j])-1 # sub-tour elimination for truck

status = problem.solve() 
print(f"status: {problem.status}, {LpStatus[problem.status]}")
print(f"objective: {problem.objective.value()}")
for var in problem.variables():
    if (problem.status == 1):
        if (var.value() !=0):
            print(f"{var.name}: {var.value()}")

import matplotlib.pyplot as plt
import numpy as np

city_coordinates = {
    0: (0, 0),
    1: (1, 2),
    2: (3, 1),
    3: (2, 4),
    4: (4, 3),
    5: (5, 0),
}

plt.figure(figsize=(8, 6))
for city, coords in city_coordinates.items():
    plt.scatter(coords[0], coords[1], color='magenta', s=100)
    plt.text(coords[0] + 0.1, coords[1] + 0.1, str(city), fontsize=12)
for i in range(row):
    for j in range(row):
        if decisionVariableX[i, j].varValue > 0.5:  # Pokud je cesta aktivní
            x_coords = [city_coordinates[i][0], city_coordinates[j][0]]
            y_coords = [city_coordinates[i][1], city_coordinates[j][1]]
            plt.plot(x_coords, y_coords, color='deeppink')

plt.title('Optimální trasa obchodního cestujícího', fontweight="bold")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### dopravní problém
```python
from pulp import *

x11 = LpVariable("x11", lowBound = 0)
x12 =  LpVariable("x12", lowBound = 0)
x13 =  LpVariable("x13", lowBound = 0)
x21 =  LpVariable("x21", lowBound = 0)
x22 =  LpVariable("x22", lowBound = 0)
x23 =  LpVariable("x23", lowBound = 0)
x31 =  LpVariable("x31", lowBound = 0)
x32 =  LpVariable("x32", lowBound = 0)
x33 =  LpVariable("x33", lowBound = 0)
x41 =  LpVariable("x41", lowBound = 0)
x42 =  LpVariable("x42", lowBound = 0)
x43 =  LpVariable("x43", lowBound = 0)
x51 =  LpVariable("x51", lowBound = 0)
x52 =  LpVariable("x52", lowBound = 0)
x53 =  LpVariable("x53", lowBound = 0)

problem = LpProblem("myProblem", LpMinimize)
problem += 100 * x11 + 75 * x12 + 40 * x13 + 20 * x21 + 75 * x22 + 80 * x23 + 80 * x31 + 15 * x32 + 60 * x33 + 70 * x41 + 45 * x42 + 50 * x43 + 30 * x51 + 85 * x52 + 90 * x53
problem += x11 + x12 + x13 == 20
problem += x21 + x22 + x23 == 30
problem += x31 + x32 + x33 == 40
problem += x41 + x42 + x43 == 35
problem += x51 + x52 + x53 == 25
problem += x11 + x21 + x31 + x41 + x51 == 50
problem += x12 + x22 + x32 + x42 + x52 == 50
problem += x13 + x23 + x33 + x43 + x53 == 50
print(problem)
status = problem.solve()
print(f"Status:\n{LpStatus[status]}\n")
print("Solution:")
for v in problem.variables():
  print(f"\t\t{v.name} = {v.varValue}")
print("\n")
print(f"Optimal solution: {problem.objective.value()}")
```

### Přiřazovací problém 
```python
from pulp import *

x11 = LpVariable("x11", cat="Binary")
x12 =  LpVariable("x12", cat="Binary")
x13 =  LpVariable("x13", cat="Binary")
x14 =  LpVariable("x14", cat="Binary")
x15 =  LpVariable("x15", cat="Binary")
x21 =  LpVariable("x21", cat="Binary")
x22 =  LpVariable("x22", cat="Binary")
x23 =  LpVariable("x23", cat="Binary")
x24 =  LpVariable("x24", cat="Binary")
x25 =  LpVariable("x25", cat="Binary")
x31 =  LpVariable("x31", cat="Binary")
x32 =  LpVariable("x32", cat="Binary")
x33 =  LpVariable("x33", cat="Binary")
x34 =  LpVariable("x34", cat="Binary")
x35 =  LpVariable("x35", cat="Binary")
x41 =  LpVariable("x41", cat="Binary")
x42 =  LpVariable("x42", cat="Binary")
x43 =  LpVariable("x43", cat="Binary")
x44 =  LpVariable("x44", cat="Binary")
x45 =  LpVariable("x45", cat="Binary")
x51 =  LpVariable("x51", cat="Binary")
x52 =  LpVariable("x52", cat="Binary")
x53 =  LpVariable("x53", cat="Binary")
x54 =  LpVariable("x54", cat="Binary")
x55 =  LpVariable("x55", cat="Binary")

problem = LpProblem("myProblem", LpMinimize)
problem += 5 * x11 + 10 * x12 + 5 * x13 + 15 * x14 + 20 * x15 + 10 * x21 + 20 * x22 + 10 * x23 + 15 * x24 + 5 * x25 + 10 * x31 + 30 * x32 + 10 * x33 + 5 * x34 + 5 * x35 + 10 * x41 + 5 * x42 + 10 * x43 + 0 * x44 + 0 * x45 + 1 * x51 + 5 * x52 + 5 * x53 + 15 * x54 + 20 * x55

problem += x11 + x12 + x13 + x14 + x15 == 1
problem += x21 + x22 + x23 + x24 + x25 == 1
problem += x31 + x32 + x33 + x34 + x35 == 1
problem += x41 + x42 + x43 + x44 + x45 == 1
problem += x51 + x52 + x53 + x54 + x55 == 1

problem += x11 + x21 + x31 + x41 + x51 == 1
problem += x12 + x22 + x32 + x42 + x52 == 1
problem += x13 + x23 + x33 + x43 + x53 == 1
problem += x14 + x24 + x34 + x44 + x54 == 1
problem += x15 + x25 + x35 + x45 + x55 == 1




print(problem)
status = problem.solve()
print(f"Status:\n{LpStatus[status]}\n")
print("Solution:")
for v in problem.variables():
  print(f"\t\t{v.name} = {v.varValue}")
print("\n")
print(f"Optimal solution: {problem.objective.value()}")
```
