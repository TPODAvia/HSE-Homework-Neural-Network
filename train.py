#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is used:")
print(device)

# from torch.utils.tensorboard import SummaryWriter
# import sys

from model import Net
from Lab1.lab1_prep import Lab1Class
from Lab2.lab2_prep import Lab2Class
from Lab3.lab3_prep import Lab3Class

lab = Lab1Class()

X, y = lab.preprocess_fit()
data_input_size, data_output_size, learning_rate, loss_method = lab.get_nn_param()
print(f"Input shape: {data_input_size} {data_output_size}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
# print(y_train)

# Convert pandas DataFrames to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

# Create TensorDataset and DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the network
model = Net(input_size=data_input_size, output_size=data_output_size)
# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")
# Move the model to the CPU/GPU
model = model.to(device)

# Define the loss function and optimizer
criterion = loss_method
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# writer = SummaryWriter()

if not os.path.exists(lab.lab_dir()):
    os.mkdir(lab.lab_dir())

print("Training...")
# Training loop
num_epochs =  50000
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_mod, target_mod = lab.output_mod(model(data), target)
        loss = criterion(output_mod,target_mod)
        loss.backward()
        optimizer.step()

        # Log the loss to TensorBoard
        # writer.add_scalar('Loss/train', loss.item(), epoch)

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        total_loss =  0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_mod, target_mod = lab.output_mod(model(data), target)
            loss = criterion(output_mod,target_mod)
            total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        print(f'Epoch: {epoch+1}, Validation Loss: {loss} {output} {target}')

        # writer.add_scalar('Loss/Validation', avg_loss, epoch)

    if epoch%500 == 0: # every 500 epoch save the model
        torch.save(model.state_dict(), f'{lab.lab_dir()}model.pth')
 