import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda

# Dane i transformacje
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = Compose([
    ToTensor(),
    Lambda(lambda x: x.view(-1))
])

full_dataset = ImageFolder(root='dataset1', transform=transform)
sample_img, _ = full_dataset[0]
input_dim = sample_img.shape[0]

n_total = len(full_dataset)
n_train = int(0.85 * n_total)
n_test = n_total - n_train
trainset, testset = random_split(full_dataset, [n_train, n_test])

# Definicja modelu z dynamicznymi hiperparametrami
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_units, activation_type):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 4)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.activation_type = activation_type

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return self.logsoftmax(x)

    def activate(self, x):
        if self.activation_type == 0:
            return torch.relu(x)
        elif self.activation_type == 1:
            return torch.tanh(x)
        elif self.activation_type == 2:
            return F.leaky_relu(x)
        else:
            return x

# Funkcje treningu i ewaluacji
def train(model, dataset, n_iter, batch_size, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device).train()
    for _ in range(n_iter):
        for imgs, targets in loader:
            optimizer.zero_grad()
            out = model(imgs.to(device))
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()

def accuracy(model, dataset):
    loader = DataLoader(dataset, batch_size=256)
    model.to(device).eval()
    correct = 0
    with torch.no_grad():
        for imgs, targets in loader:
            pred = model(imgs.to(device)).argmax(dim=1)
            correct += (pred == targets.to(device)).sum().item()
    return correct / len(dataset)

# Genetyczny kod
def generate_individual():
    return [
        random.uniform(0.0001, 0.1),         # learning rate
        random.choice([64, 128, 256, 512]),  # batch size
        random.choice([64, 128, 256, 512]),  # hidden layer size
        random.randint(0, 2),                # activation type: 0=ReLU, 1=Tanh, 2=LeakyReLU
        random.randint(5, 20)                # number of epochs
    ]

def mutate(individual):
    i = random.randint(0, len(individual)-1)
    if i == 0:
        individual[0] = random.uniform(0.0001, 0.1)
    elif i == 1:
        individual[1] = random.choice([64, 128, 256, 512])
    elif i == 2:
        individual[2] = random.choice([64, 128, 256, 512])
    elif i == 3:
        individual[3] = random.randint(0, 2)
    elif i == 4:
        individual[4] = random.randint(5, 20)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-2)
    return parent1[:point] + parent2[point:]

def evaluate(individual):
    lr, batch_size, hidden, act, n_iter = individual
    model = MyModel(input_dim=input_dim, hidden_units=hidden, activation_type=act)
    train(model, trainset, n_iter=n_iter, batch_size=batch_size, lr=lr)
    return accuracy(model, testset)

# Algorytm genetyczny
def genetic_algorithm(n_generations=5, population_size=6):
    population = [generate_individual() for _ in range(population_size)]
    for gen in range(n_generations):
        print(f"\n Generacja {gen+1}")
        scored_population = [(ind, evaluate(ind)) for ind in population]
        scored_population.sort(key=lambda x: x[1], reverse=True)
        print("Najlepszy wynik:", scored_population[0][1]*100, "%")
        print("Chromosom:", scored_population[0][0])

        # Selekcja najlepszych (elitism + krzyżowanie)
        survivors = [ind for ind, _ in scored_population[:2]]  # top 2
        children = []
        while len(children) < population_size - len(survivors):
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1, p2)
            mutate(child)
            children.append(child)
        population = survivors + children

    # Najlepszy osobnik
    best_individual = max(population, key=lambda ind: evaluate(ind))
    print("\n Najlepszy zestaw parametrów:", best_individual)
    return best_individual

if __name__ == "__main__":
    best = genetic_algorithm(n_generations=5, population_size=6)
