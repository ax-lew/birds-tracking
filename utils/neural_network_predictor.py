import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

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
        self.linear1 = nn.Linear(10, 8, bias=True)
        self.linear2 = nn.Linear(8, 4, bias=True)
        self.linear3 = nn.Linear(4, 2, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)             
        return x



class NeuralNetworkPredictor:
    """docstring for NeuralNetworkPredictor"""
    def __init__(self, l_rate = 0.00000001, epochs = 400000, x_scaler=False, y_scaler=False):
        self.model = Model()
        self.criterion = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr = l_rate) 
        self.epochs = epochs
        self.l_rate = l_rate
        self.y_scaler = StandardScaler() if y_scaler else None
        self.x_scaler = StandardScaler() if x_scaler else None

    def clear(self):
        self.model = Model()
        self.criterion = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr = self.l_rate) 
        self.y_scaler = StandardScaler() if self.y_scaler else None
        self.x_scaler = StandardScaler() if self.x_scaler else None

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def open_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def preprocess_data(self, data):
        data = np.append(data, (data[:,0]*data[:,1]).reshape((len(data),1)), axis=1)
        data = np.append(data, (data[:,0]*data[:,2]).reshape((len(data),1)), axis=1)
        data = np.append(data, (data[:,0]*data[:,3]).reshape((len(data),1)), axis=1)
        data = np.append(data, (data[:,1]*data[:,2]).reshape((len(data),1)), axis=1)
        data = np.append(data, (data[:,1]*data[:,3]).reshape((len(data),1)), axis=1)
        data = np.append(data, (data[:,2]*data[:,3]).reshape((len(data),1)), axis=1)

        return data

    def fit(self, data, target):
        data = self.preprocess_data(data.copy())
        data = np.array(data.astype(float))
        target = np.array(target.astype(float))
        
        if self.x_scaler is not None:
            data = self.x_scaler.fit_transform(data)
        if self.y_scaler is not None:
            target = self.y_scaler.fit_transform(target)

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
            if epoch > 1000 and random.randint(0, 1000) == 1: print('epoch {}, loss {}'.format(epoch,loss.item()))

    def predict(self, data):
        data = self.preprocess_data(data.copy())
        data = np.array(data.astype(float))
        if self.x_scaler is not None:
            data = self.x_scaler.transform(data)
        test = Variable(torch.from_numpy(data))
        test = test.clone().detach().float()
        predicted = self.model.forward(test).data.numpy()
        if self.y_scaler is not None:
            predicted = self.y_scaler.inverse_transform(predicted)
        return predicted
        
