import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(Net, self).__init__()
        
        self.linear_relu1 = nn.Linear(input_size,  32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear_relu2 = nn.Linear(32,  256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.linear_relu3 = nn.Linear(256,  256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # self.linear_relu4 = nn.Linear(256,  256)
        # self.bn4 = nn.BatchNorm1d(256)
        # self.dropout4 = nn.Dropout(dropout_rate)
        
        self.linear5 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear_relu1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.linear_relu2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.linear_relu3(x)))
        x = self.dropout3(x)
        
        # x = F.relu(self.bn4(self.linear_relu4(x)))
        # x = self.dropout4(x)
        
        x = self.linear5(x)
        return x