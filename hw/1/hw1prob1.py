import torch
from enum import Enum


class DatasetKind(Enum):
    OR = 0
    AND = 1
    XOR = 2


class Perceptron(torch.nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.fc(x)
        output = self.sigmoid(output)
        return output


class Feedforward(torch.nn.Module):
    def __init__(self, input_size):
        hidden_size = 10
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class Dataset:
    def __init__(self, dataset_kind):
        self.x, self.y = self.create_dataset(dataset_kind)

    @staticmethod
    def create_dataset(dataset_kind):
        x = []
        y = []
        if dataset_kind == DatasetKind.XOR:
            for i in range(0, 2):
                for j in range(0, 2):
                    x.append([i, j])
                    if dataset_kind == DatasetKind.XOR:
                        y.append(i ^ j)
        else:
            for i in range(0, 2):
                for j in range(0, 2):
                    for k in range(0, 2):
                        x.append([i, j, k])
                        if dataset_kind == DatasetKind.OR:
                            y.append(i | j | k)
                        elif dataset_kind == DatasetKind.AND:
                            y.append(i & j & k)

        return torch.Tensor(x), torch.Tensor(y)


def single_layer_train(dataset, model_kind, epoch):
    input_size = dataset.x.size(1)
    model = model_kind(input_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.eval()
    y_pred = model(dataset.x)
    before_train = criterion(y_pred.squeeze(), dataset.y)
    print('Test loss before training', before_train.item())

    model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        y_pred = model(dataset.x)
        loss = criterion(y_pred.squeeze(), dataset.y)

        # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        loss.backward()
        optimizer.step()

    model.eval()
    y_pred = model(dataset.x)
    after_train = criterion(y_pred.squeeze(), dataset.y)
    print('Test loss after Training', after_train.item())
    print('Input: ', dataset.x)
    print('Target: ', dataset.y)
    print('Predicted: ', y_pred)


if __name__ == '__main__':
    dataset = Dataset(DatasetKind.XOR)
    # single_layer_train(dataset, Feedforward, 10000)
    single_layer_train(dataset, Perceptron, 50000)
