import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt 

# 0) prepare dataset
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, random_state=13)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32).reshape(-1, 1))

# 1) model 
input_size = X.shape[1]
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3) Training loop 
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

# plot
prediction = model(X).detach().numpy()
plt.scatter(X_numpy[:,0], y_numpy, color='red')
plt.plot(X_numpy[:,0], prediction, color='blue')
plt.show()
