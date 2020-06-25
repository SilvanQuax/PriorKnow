import tqdm
from pytorch_networks import *
from torch.autograd import Variable
import torch.nn.functional as F
import copy

def train_network(model,optimizer,train_iter,test_iter=None,n_epochs=None,n_samples=None,cutoff=1,act_reg=False):
    train_loss = np.zeros(n_epochs)
    accuracy = np.zeros(n_epochs)

    test_loss = np.zeros(n_epochs)
    if isinstance(model, Classifier):
        dat_id = 1
    elif isinstance(model, Regressor):
        dat_id = 1

    for epoch in tqdm.tqdm(xrange(n_epochs)):

        n=0
        loss = 0
        if test_iter is not None:
            for data in test_iter:
                if isinstance(model, Classifier):
                    data[dat_id] =data[dat_id].astype('int64')
                n+=1

                # train step
                _,accuracy[epoch] = model(Variable(torch.from_numpy(data[0])),Variable(torch.from_numpy(data[dat_id])))
                break

        for data in train_iter:
            if isinstance(model, Classifier):
                data[dat_id] =data[dat_id].astype('int64')
            n+=1

            # train step
            loss += model(Variable(torch.from_numpy(data[0])),Variable(torch.from_numpy(data[dat_id])))[0]

            if n == 1:
                M0 = Variable(torch.from_numpy(np.zeros(model.predictor.h.shape).astype('float32')))
            if act_reg == True:
                loss += 150*F.mse_loss(model.predictor.h, M0)

            # if n==100:
            #     optimizer.param_groups[0]['lr'] = 0.0001

            # if n == 1:
            #     M0 = Variable(np.zeros(model.predictor.h.shape).astype('float32'))
            # if act_reg ==True:
            #     loss += F.mean_squared_error(model.predictor.h,M0)
            train_loss[epoch] += loss.data

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            loss = 0
            #model.predictor.l1.W.W.data *= mask.astype('float32')
            if n == n_samples:
                break


        train_loss[epoch] /= n_samples
        print(['loss = %.2f' % train_loss[epoch]])

    return train_loss, accuracy

class tester(object):
    # generate posterior values for test data

    def __init__(self,model):
        self.model = model

    def run(self,test_iter,out = 1):
        for data in test_iter:
            if isinstance(self.model, Classifier):
                y = F.softmax(self.model.predictor(Variable(torch.from_numpy(data[0]))))
                # y = F.sigmoid(self.model.predictor(Variable(torch.from_numpy(data[0]))))
                # y = self.model.predictor(Variable(torch.from_numpy(data[0])))

            elif isinstance(self.model, Regressor):
                y = self.model.predictor(Variable(torch.from_numpy(data[0])))


            self.y=y
            self.test_posterior = y.data.numpy()
            self.real_posterior = data[4]
            self.real_posterior2 = data[5]
            self.H = self.model.predictor.h.data.numpy()
            self.G = data[2]
            self.R = data[0]
            self.S = data[3]
            self.C = data[1]
            if len(data)>6:
                self.SIG = data[6]

class bp_tester(object):
    # generate posterior values for test data

    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
        self.predictors = []
    def run(self,test_iter,out = 1):
        for data in test_iter:
            if isinstance(self.model, Classifier):
                data[1] = data[1].astype('int64')
                loss = self.model(Variable(torch.from_numpy(data[0])), Variable(torch.from_numpy(data[1])))[0]
                # y = F.sigmoid(self.model.predictor(Variable(torch.from_numpy(data[0]))))
                # y = self.model.predictor(Variable(torch.from_numpy(data[0])))
                self.optimizer.zero_grad()
                loss.backward()

            elif isinstance(self.model, Regressor):
                y = self.model.predictor(Variable(torch.from_numpy(data[0])))

            self.predictors.append(np.mean(np.abs((self.model.predictor.l1.weight.grad.numpy()))))
