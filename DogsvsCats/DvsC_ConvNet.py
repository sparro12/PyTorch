import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from matplotlib import style

# change to True if the data (i.e. npy file needs to be built)
REBUILD_DATA = False
TRAIN_MODEL = False
MODEL_NAME = f"model-{int(time.time())}" # create model name based on current time

rebuild = input("Build the data? (y/n): ")
while rebuild != 'y' and rebuild != 'n':
    rebuild = input("Enter y or n: ")

if rebuild == 'y':
    REBUILD_DATA = True
elif rebuild == 'n':
    REBUILD_DATA = False

train_model = input("Train the model? (y/n): ")
while train_model != 'y' and train_model != 'n':
    train_model = input("Enter y or n: ")

if train_model == 'y':
    TRAIN_MODEL = True
elif rebuild == 'n':
    TRAIN_MODEL = False


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

# train model
def train(net, train_X, train_Y, test_X, test_Y):
    BATCH_SIZE = 100
    EPOCHS = 8

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
                batch_Y = train_Y[i:i+BATCH_SIZE]

                acc, loss = fwd_pass(batch_X, batch_Y, train=True)

                if i % 50 == 0:
                    val_acc, val_loss = test(test_X, test_Y, size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")

# find accuracy of Model with test data
def test(test_X, test_Y, size=32):
    X, Y = test_X[:size], test_Y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50), Y)
    return val_acc, val_loss

def fwd_pass(X, Y, train=False):

    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, Y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, Y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss

# create graph of losses and accuracy over time
def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []
    epochs = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))
            epochs.append(float(epoch))


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times,losses, label="loss")
    ax2.plot(times,val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()


if __name__ == "__main__" :
    if TRAIN_MODEL:
        # initialze neural network
        net = Net()
        print(net)

        # builds training data
        if REBUILD_DATA:
            dogsvscats = DogsvsCats()
            dogsvscats.make_training_data()

        training_data = np.load("training_data.npy", allow_pickle=True)
        print(len(training_data))

        # initialze optimizer and loss function to train model
        optimizer = optim.Adam(net.parameters(), lr = 0.001)
        loss_function = nn.MSELoss()

        X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) # image
        X = X/255.0
        Y = torch.Tensor([i[1] for i in training_data]) # one hot vector for type

        VAL_PCT = 0.1  # reserved percentage of data for validation
        val_size = int(len(X)*VAL_PCT)
        print("val_size =", val_size)

        train_X = X[:-val_size]
        train_Y = Y[:-val_size]

        test_X = X[-val_size:]
        test_Y = Y[-val_size:]

        print("train_X =", len(train_X))
        print("test_X =", len(test_X))

        train(net, train_X, train_Y, test_X, test_Y)
        val_acc, val_loss = test(test_X, test_Y, size=100)
        print(val_acc, val_loss)

    style.use("ggplot")
    create_acc_loss_graph("model-1608524563") # create graph
