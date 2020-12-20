import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor()
                       ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

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

net = Net()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(3): # 3 full passes over the data
    for data in trainset:  # data is a batch of data
        X,Y = data # X is the batch of features, y is the batch of targets
        net.zero_grad() # sets gradients to 0 for loss calc. Most likely occurs every step
        output = net(X.view(-1, 784)) # pass in the reshaped batch
        loss = F.nll_loss(output, Y) # calc and grab the loss value
        loss.backward() # apply this loss backwards thru the network's parameters
        optimizer.step() # attempt to optimize weights to account for loss/gradients
    print(loss)

# Save Model
PATH = "MNIST_model.pt"
torch.save(net, PATH)
