import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# change to True if the data (i.e. npy file needs to be built)
REBUILD_DATA = False

rebuild = input("Build the data? (y/n): ")
while rebuild != 'y' and rebuild != 'n':
    rebuild = input("Enter y or n: ")

if rebuild == 'y':
    REBUILD_DATA = True
elif rebuild == 'n':
    REBUILD_DATA = False

class DogsvsCats():
    img_size = 50
    cats = "PetImages/Cat"
    dogs = "PetImages/Dog"
    testing = "PetImages/Testing"
    labels = {cats: 0, dogs: 1}
    training_data = []

    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.labels:
            print(label)
            for image in tqdm(os.listdir(label)):
                if "jpg" in image:
                    try:
                        path = os.path.join(label, image)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        self.training_data.append([np.array(img), np.eye(2)[self.labels[label]]])

                        if label == self.cats:
                            self.catcount += 1
                        elif label == self.dogs:
                            self.dogcount += 1

                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

# builds training data
if REBUILD_DATA:
    dogsvscats = DogsvsCats()
    dogsvscats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

class Net(nn.Module):
    def __init__(self):
        super().__init__() # initializes parent class: nn.Module
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel/window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, 64 output channels, 5x5 kernel/window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)


net = Net()

optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) # image
X = X/255.0
Y = torch.Tensor([i[1] for i in training_data]) # one hot vector for type

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print("val_size =", val_size)

train_X = X[:-val_size]
train_Y = Y[:-val_size]

test_X = X[-val_size:]
test_Y = Y[-val_size:]

print("train_X =", len(train_X))
print("test_X =", len(test_X))

BATCH_SIZE = 100
EPOCHS = 5

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_Y = train_Y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_Y)
        loss.backward()
        optimizer.step()

print(loss)

# find accuracy of Model with test data
correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_Y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy", round(correct/total, 3))
