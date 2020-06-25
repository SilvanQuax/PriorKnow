import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tasks import *
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(50,200)
        self.fc2 = nn.Linear(200, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x)

import torch.optim as optim

def criterion(out, label):
    loss = F.cross_entropy(out,label)
    return loss

priors = [[0.25,0.75],[0.33,0.67],[0.5,0.5],[0.67,0.33],[0.75,0.25]]

params={}

# training parameters
params['n_epochs'] = 50
params['N_input'] = 50
params['N_hidden'] = 200
params['N_train'] = 100000
params['N_test'] = 20000
params['N_class'] = 2
params['N_cond'] = 1

# data parameters
params['means']=[-5.0, 5.0]
params['sigmas']=[25.0, 25.0]
params['sig_phi']=10.0
params['gain']=(np.array([0.37,0.90,1.81,2.82,3.57,4.00])*0.8).tolist()#[0.37,0.90,1.81,2.82,3.57,4.00]*0.4 #*0.4
params['prior']=priors[0]

locals().update(params)

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

train_data = classification_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,original=False)
data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]
for epoch in range(100):
    for i, data2 in enumerate(train_data):
        X = data2[0]
        Y = data2[1]
        X, Y = Variable(torch.from_numpy(np.expand_dims(X,0)), requires_grad=True), Variable(torch.LongTensor([int(Y)]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0):
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))