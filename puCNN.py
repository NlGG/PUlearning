import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from densratio import densratio


class NeuralNet(chainer.Chain):
    def __init__(self):
        super().__init__(
        	self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)
        )

    def __call__(self, x0, x1, t, pi):
        n0 = len(x0)
        n1 = len(x1)
        
        x1 = Variable(x1)
        x0 = Variable(x0)
        
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        gp = self.fc8(h)
        
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        gu = self.fc8(h)
                
        Rpplus = F.sum(1/(1+F.exp(gp)))/n1
        Rpminus = F.sum(1/(1+F.exp(-gp)))/n1
        Ruminus = F.sum(1/(1+F.exp(-gu)))/n0
        
        J0 = Ruminus - pi*Rpminus
        
        if J0.data < 0:
            return - J0
        else:
            return 2*pi*Rpplus + Ruminus - pi
        #J1 + (1/n)*F.sum(F.sigmoid(-h8))
    
    def check_accuracy(self, xs, ts):
        x = Variable(xs)
        
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        ys = F.sigmoid(h)
        
        ys = ys.data.T[0]
        
        ans = ys
        ans[ys >= 0.5] = 1
        ans[ys < 0.5] = 0
        
        acc = (ts - ans)**2
                
        return np.mean(acc)

train, test = chainer.datasets.get_mnist()
xs, ts = train._datasets
xs = xs
ts = ts

ts[ts == 1] = 5
ts[ts%3 == 0] = 1
ts[(ts%3 != 0)&(ts != 1)] = 0

truets = np.copy(ts)

n = len(ts)
rv = np.random.uniform(0, 1, n)

ts[(ts == 1)&(rv < 0.5)] = 0

xs0 = xs
xs1 = xs[ts == 1]

txs, tts = test._datasets
tts[tts == 1] = 5
tts[tts%3 == 0] = 1
tts[(tts%3 != 0)&(tts != 1)] = 0

bm = 100

pi = 0.9
pi_1 = 100
pi_2 = 100

count = 0

temp = np.copy(xs0)

for i in range(200):
    for j in range(600):
        model.zerograds()
        x0 = xs0[(j * bm):((j + 1) * bm)]
        t = ts[(j * bm):((j + 1) * bm)]
        loss = model(x0, t, pi)
        loss.backward()
        optimizer.update()
        
    pi_2 = pi_1
    pi_1 = pi
    
    pi = 0
    for j in range(100):
        np.random.shuffle(temp)
        x0 = temp[:50000]
        ptemp = model.prob(x0)
        pi += ptemp.data
    
    pi = pi/100
    
    if pi_2 - pi_1 < pi_1 - pi:
        pi = pi_1
        
    pis[i] = pi            
    print(pi)

        
    loss0 = model.check_accuracy(xs, truets)
    train_loss[i] = loss0
    print('train_loss', loss0)
    loss1 = model.check_accuracy(txs, tts)
    test_loss[i] = loss1
    print('test_loss', loss1)
