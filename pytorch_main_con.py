import torch
import numpy as np
import json
import mkl
mkl.set_num_threads(10)
import torch.functional as F


from tasks import *
from pytorch_networks import *
from tools import *
from pytorch_trainer import *

priors = [100.0, 50.0, 25.0, 10.0, 5.0]

trainl = []

for ii in xrange(1,6):
    task = 'con_'+str(ii)

    #param_loc = '/vol/ccnlab1/silqua/prirep/models/pytorch/param_GNH200_3' + task
    #model_loc = '/vol/ccnlab1/silqua/prirep/models/pytorch/model_GNH200_3' + task
    param_loc = '/vol/ccnlab1/silqua/prirep/models/pytorch/param_GNH204_rev' + task
    model_loc = '/vol/ccnlab1/silqua/prirep/models/pytorch/model_GNH204_rev' + task

    params={}

    # training parameters
    params['n_epochs'] = 100
    params['N_input'] = 50
    params['N_hidden'] = 204
    params['N_train'] = 200000
    params['N_test'] = 20000
    params['N_cond'] = 1

    # data parameters
    params['mean']=0.0
    params['sigma']=priors[ii-1]
    params['sig_phi']=10.0
    params['gain']=(np.array([0.37,0.90,1.81,2.82,3.57,4.00])*0.8).tolist()#[0.37,0.90,1.81,2.82,3.57,4.00]*0.4 #*0.4
    # params['gain']=(np.linspace(0.2,1.0,6)).tolist()#[0.37,0.90,1.81,2.82,3.57,4.00]*0.4 #*0.4
    # params['gain']=(np.linspace(0.5,4,6)).tolist()#[0.37,0.90,1.81,2.82,3.57,4.00]*0.8 #*0.4


    with open(param_loc+'.json','w') as fp:
        json.dump(params, fp, indent=0 )

    locals().update(params)
    # train_data = conditional_classification_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,original=False)
    # test_data = conditional_classification_task(mu=means,sig2=sigmas,n_samples=N_test, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,test=True,original=False)

    # train_data = classification_task(mu=means,sig2=sigmas,n_samples=N_train, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,original=False)
    # test_data = classification_task(mu=means,sig2=sigmas,n_samples=N_test, sig2N = sig_phi, g=gain, n_input=N_input, pCin=prior,test=True,original=False)

    train_data = continuous_task2(n_samples=N_train, n_input=N_input, sig2N=sig_phi, muP=mean, sig2P=sigma, g=gain, test=False)
    test_data = continuous_task2(n_samples=N_test, n_input=N_input, sig2N=sig_phi, muP=mean, sig2P=sigma, g=gain, test=True)

    # train_data = continuous_task(n_samples=N_train, g=gain, sig2N = sig_phi, n_input=N_input,original=False)
    # test_data = continuous_task(n_samples=N_test, g=gain, sig2N = sig_phi, n_input=N_input, test=True,original=False)

    # define data iterators
    train_iter = RandomIterator(train_data, batch_size=10)
    test_iter = FixedIterator(test_data, batch_size=N_test*len(gain)*N_cond)

    # define model
    # model = Classifier(MLP(N_hidden, 2, no_bias = True))
    model = Regressor(MLP_nobias(N_input,N_hidden, 1, bias = False))

    # Setup an optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0002,weight_decay=0.001)

    #optimizer.add_hook(chainer.optimizer.Lasso(0.01))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))

    # test = tester(model)

    train_loss, accuracy = train_network(model,optimizer,train_iter,n_epochs=n_epochs,n_samples=1000,act_reg=False)
    # accs.append(accuracy)
    trainl.append(train_loss)


    torch.save(model.state_dict(), model_loc)

1




