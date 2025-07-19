# Introduction

The goal of our university project was to design and implement an autonomous forklift capable of navigating a warehouse environment,
locating designated chests, and transporting them to specified drop‑off shelfs without human intervention. The system integrates
multiple AI and search algorithms to handle path planning and decision making.

# Program Workflow

## 1. Shelf Population
At the start of each task, target chests are placed on designated shelves within the warehouse grid.

## 2. Chest Selection
A trained neural network evaluates all visible chests and selects the next one to retrieve.
This choice is based on learned criteria such as priority level, type of chest, distance and waiting time.

## 3. Path Planning to Chest
Once the target chest is chosen, the forklift invokes the A* algorithm to compute an optimal collision‑free 
path from its current position to the chest’s location on the shelf.

## 4. Dynamic Drop‑off Planning
After picking up the chest, the forklift runs a secondary A* planning pass to determine the best delivery point. 
During this step, it analyzes the planned route for potential slowdowns or moving obstacles.

## 5. Delivery and Loop
The forklift transports the chest to the chosen drop‑off location, unloads it, and then re‑starts the entire process—scanning for the next chest.

# Algorithms Employed
- **Uninformed Search:** Breadth‑First Search (BFS)  
- **Informed Search:** A*  
- **Decision Trees:** ID3 algorithm  
- **Neural Networks**  
- **Genetic Algorithms**
