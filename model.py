import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = Compose([
    ToTensor(),
    Lambda(lambda x: x.view(-1))
])

full_dataset = ImageFolder(root='dataset1', transform=transform)

sample_img, _   = full_dataset[0]
input_dim       = sample_img.shape[0]

n_total = len(full_dataset)
n_train = int(0.85 * n_total)
n_test  = n_total - n_train
trainset, testset = random_split(full_dataset, [n_train, n_test])

class MyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(64, 4)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.logsoftmax(self.fc2(x))
        return x

model = MyModel(input_dim=input_dim)

def train(model, dataset, n_iter=18, batch_size=128):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.048780882415749016)
    criterion = nn.NLLLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device).train()
    start = time.time()
    for _ in range(n_iter):
        for imgs, targets in loader:
            optimizer.zero_grad()
            out = model(imgs.to(device))
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()
    return time.time() - start

def accuracy(model, dataset):
    loader = DataLoader(dataset, batch_size=256)
    model.to(device).eval()
    correct = 0
    with torch.no_grad():
        for imgs, targets in loader:
            pred = model(imgs.to(device)).argmax(dim=1)
            correct += (pred == targets.to(device)).sum().item()
    return correct / len(dataset)

if __name__ == "__main__":
    elapsed = train(model, trainset, n_iter=10, batch_size=256)
    print(f"Czas trenowania: {elapsed:.2f}s")

    acc = accuracy(model, testset)
    print(f"Dokładność na zbiorze testowym: {acc*100:.2f}%")
    torch.save(model.state_dict(), "model_scratch_new_abs_best.pth")