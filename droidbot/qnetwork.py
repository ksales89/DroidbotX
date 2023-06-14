import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state.view(-1, self.fc1.in_features * torch.numel(state))))
        #x = torch.relu(self.fc1(state.view(-1, self.fc1.in_features * state[0])))#considera uma dimensão extra na entrada dos dados
        #x = torch.relu(self.fc1(state.view(-1, self.fc1.in_features).squeeze(dim=1))) # remove a dimensão adicional do tensor state, se existir
        #x = torch.relu(self.fc1(state.view(-1, self.fc1.in_features * state[0] * state[1])))
        #x = torch.relu(self.fc1(state.view(-1, self.fc1.in_features * state[0] * state[1] * state[2]))) #adiciona um novo fator state_size[2] ao redimensionar o tensor de entrada, levando em consideração a terceira dimensão dos dados.
        #x = torch.relu(self.fc1(state.view(-1, self.fc1.in_features * torch.prod(state.size()[1:]))))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    
""" class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state.view(-1, self.fc1.in_features)))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values """
