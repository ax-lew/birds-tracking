import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

class LinearClassificationModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearClassificationModel, self).__init__() 
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.logprob = nn.LogSoftmax(dim=1)                 # -Log(Softmax probability).


    def forward(self, x):
        x = self.linear(x)
        x = self.logprob(x)
        return x


class NNClassifier(object):
    """docstring for NeuralNetworkPredictor"""
    def __init__(self, l_rate = 0.01, epochs = 1000):
        super(NNClassifier, self).__init__()
        self.model = LinearClassificationModel(4, 543)
        self.criterion = nn.NLLLoss()
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

        for epoch in range(epochs):

            epoch +=1
            #increase the number of epochs by 1 every time

            inputs = Variable(torch.Tensor(data), requires_grad=False)
            labels = Variable(torch.Tensor(target).long(), requires_grad=False)


            #clear grads as discussed in prev post
            optimiser.zero_grad()
            #forward to get predicted values
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()# back props
            optimiser.step()# update the parameters
            if k_index == 0 and epoch < 1000: print('epoch {}, loss {}'.format(epoch,loss.item()))
            if k_index == 0 and epoch > 1000 and random.randint(0, 100) == 1: print('epoch {}, loss {}'.format(epoch,loss.item()))

    def predict(self, data):
        data = np.array(data.astype(float))
        test = Variable(torch.from_numpy(data))
        test = test.clone().detach().float()
        predicted = self.model.forward(test).data.numpy()
        return predicted
        
