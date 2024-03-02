import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(Net, self).__init__()
        
        self.linear_relu1 = nn.Linear(input_size,  128)
        self.linear_relu2 = nn.Linear(128,  256)
        self.linear_relu3 = nn.Linear(256,  256)
        self.linear_relu4 = nn.Linear(256,  256)
        self.linear5 = nn.Linear(256, output_size)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        
        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)
        y_pred = self.dropout1(y_pred)  # Apply dropout after ReLU

        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)
        # y_pred = self.dropout2(y_pred)  # Apply dropout after ReLU

        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)
        # y_pred = self.dropout2(y_pred)  # Apply dropout after ReLU

        y_pred = self.linear_relu4(y_pred)
        y_pred = nn.functional.relu(y_pred)
        # y_pred = self.dropout2(y_pred)  # Apply dropout after ReLU

        y_pred = self.linear5(y_pred)
        return y_pred  # Return the final output
