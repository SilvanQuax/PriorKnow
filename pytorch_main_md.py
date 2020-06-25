import torch
import numpy as np
import json
import mkl
mkl.set_num_threads(10)
import torch.functional as F
# import matplotlib.pyplot as plt
# plt.ioff()
# import scipy.stats as sst

from tasks import *
from pytorch_networks import *
from tools import *
from pytorch_trainer import *
priors = [[0.25,0.75],[0.33,0.67],[0.5,0.5],[0.67,0.33],[0.75,0.25]]

accs = []
trainl = []
for ii in xrange(1,6):
    task = 'class_'+str(ii)

    param_loc = '/vol/ccnlab1/silqua/prirep/models/pytorch/param_NH200_mnist2_rev' + task
    model_loc = '/vol/ccnlab1/silqua/prirep/models/pytorch/model_NH200_mnist2_rev' + task


    params={}

    # training parameters
    params['n_epochs'] = 100
    params['N_input'] = 49
    params['N_hidden'] = 200
    params['N_train'] = 3000
    params['N_test'] = 20000
    params['N_class'] = 2
    params['N_cond'] = 1

    # data parameters
    params['means']=[-5.0, 5.0]
    # params['sigmas']=[25.0, 25.0]
    # params['sig_phi']=10.0
    params['sigmas'] = [25.0, 25.0]
    params['sig_phi'] = 10.0
    # params['gain']=(np.array([0.37,0.90,1.81,2.82,3.57,4.00])*0.8).tolist()#[0.37,0.90,1.81,2.82,3.57,4.00]*0.4 #*0.4
    params['gain']=(np.linspace(0.5,4.0,6)).tolist()#[0.37,0.90,1.81,2.82,3.57,4.00]*0.4 #*0.4

    params['prior']=priors[ii-1]
    #
    with open(param_loc+'.json','w') as fp:
        json.dump(params, fp, indent=0 )

    locals().update(params)

    #train_data = md_classification_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,original=False)
    train_data = mnist_classification_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,original=False)

    # test_data = classification_task(mu=means,sig2=sigmas,n_samples=N_test, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,test=True,original=False)

    # train_data = conditional_classification_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,original=False)
    # train_data = continuous_class_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,original=False)

    # define data iterators
    train_iter = RandomIteratorMnist(train_data, batch_size=10)
    # test_iter = RandomIterator(train_data, batch_size=4000)
    # test_iter = FixedIterator(train_data, batch_size=N_test*len(gain)*N_class*N_cond)

    # define model
    # model = MLP(N_input,N_hidden, 1, bias = False)
    # model = Regressor(MLP(N_input,N_hidden, 1, bias = False))
    # model = Classifier(MLPCUE(N_input,N_hidden, 2, bias = False))
    model = Classifier(MLP2(N_input,N_hidden, 2, bias = False))

    # Setup an optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0002,weight_decay=0)

    #optimizer.add_hook(chainer.optimizer.Lasso(0.01))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))

    # test = tester(model)

    train_loss, accuracy = train_network(model,optimizer,train_iter,n_epochs=n_epochs,n_samples=1000,act_reg=False)
    accs.append(accuracy)
    trainl.append(train_loss)


    torch.save(model.state_dict(), model_loc)

    # # for param in model.predictor.parameters():
    # #     param.requires_grad = False
    # #
    # # model.predictor.l2 = nn.Linear(N_hidden,2,bias=False)
    # # model = Classifier(model.predictor)
    # # optimizer = torch.optim.Adam(model.predictor.l2.parameters(),lr=0.0001)
    # #
    # # train_loss = train_network(model,optimizer,train_iter,None,n_epochs,n_samples=1000,act_reg=False)
    #
    # model.predictor.l1.weight.requires_grad = False
    # model.predictor.l2.weight.requires_grad = False
    #
    # # optimizer = torch.optim.Adam(model.predictor.parameters(),lr=0.0001)
    #
    # train_data = classification_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=[0.75,0.25],original=False)
    # # test_data = classification_task(mu=means,sig2=sigmas,n_samples=N_test, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,test=True,original=False)
    #
    # # define data iterators
    # train_iter = RandomIterator(train_data, batch_size=10)
    #
    # train_loss = train_network(model,optimizer,train_iter,None,n_epochs,n_samples=1000,act_reg=False)
    # params['prior'] = [0.75,0.25]
    # with open(param_loc+ '_2'+'.json','w') as fp:
    #     json.dump(params, fp, indent=0 )
    #
    # torch.save(model.state_dict(), model_loc + '_2')
1
# test.run(test_iter)

# Q = extract_quantities(test,gain,N_test)
