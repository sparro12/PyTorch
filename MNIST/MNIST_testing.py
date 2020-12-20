import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor()
                       ]))

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Load
PATH = "MNIST_model.pt"
net = torch.load(PATH)
net.eval()

# find accuracy of neural net by testing it
correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, Y = data
        output = net(X.view(-1,784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == Y[idx]:
                correct += 1
            total +=1

print("Accuracy: ", round(correct/total, 3))

print(torch.argmax(net(X[0].view(-1,784))[0]))
plt.imshow(X[0].view(28,28))
plt.show()
