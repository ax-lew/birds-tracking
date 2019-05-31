import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__() 
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        
    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return out


class SingleHiddenModel(nn.Module):

    def __init__(self, input_dim, hidden_list, output_dim):

        super(SingleHiddenModel, self).__init__() 
        # Calling Super Class's constructor
        self.linear1 = nn.Linear(input_dim, hidden_dim[0], bias=True)
        self.act = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim[-1], output_dim, bias=True)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)
        return out



class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__() 
        # Calling Super Class's constructor
        self.linear1 = nn.Linear(4, 8, bias=True)
        self.linear2 = nn.Linear(8, 4, bias=True)
        self.linear3 = nn.Linear(4, 2, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)             
        return x



class NNClassifier(object):
    """docstring for NeuralNetworkPredictor"""
    def __init__(self, l_rate = 0.00000001, epochs = 400000):
        super(NNClassifier, self).__init__()
        self.model = Model()
        self.criterion = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr = l_rate) 
        self.epochs = epochs

    def clear(self):
        self.__init__()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def open_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def fit(self, data, target):
        data = np.array(data.astype(float))
        target = np.array(target.astype(float))

        for epoch in range(self.epochs):

            epoch +=1
            #increase the number of epochs by 1 every time
            inputs = Variable(torch.from_numpy(data))
            labels = Variable(torch.from_numpy(target))

            inputs = torch.tensor(inputs, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)

            #clear grads
            self.optimiser.zero_grad()
            #forward to get predicted values
            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()# back props
            self.optimiser.step()# update the parameters
            if epoch < 1000: print('epoch {}, loss {}'.format(epoch,loss.item()))
            if epoch > 1000 and random.randint(0, 100) == 1: print('epoch {}, loss {}'.format(epoch,loss.item()))

    def predict(self, data):
        data = np.array(data.astype(float))
        test = Variable(torch.from_numpy(data))
        test = test.clone().detach().float()
        predicted = self.model.forward(test).data.numpy()
        return predicted
        
