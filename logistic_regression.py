import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0) prepare dataset
bc= datasets.load_breast_cancer()
X,Y=bc.data,bc.target
print(X.shape)
print(Y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=13)

# scaling 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32)).reshape(-1,1)
y_test=torch.from_numpy(y_test.astype(np.float32)).reshape(-1,1)

print(X_train.shape)
print(y_train.shape)

# model
# f =w*x+b sigmoid at end
class LogisticRegression(nn.Module):
    def __init__(self,n_input):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input,1)
    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))

model = LogisticRegression(X_train.shape[1])

#loss and optimizer
criterion = nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

# 3) training
epochs=100
for epoch in range(epochs):
    # forward pass 
    ypred=model(X_train)
    loss=criterion(ypred,y_train)

    #backward pass
    loss.backward()

    #updates
    optimizer.step()

    # empty grad
    optimizer.zero_grad()

    if(epoch+1)%10==0:
        print(f"epoch: {epoch+1},loss={loss.item():.4f}")

with torch.no_grad():
    y_pred=model(X_test)
    y_pred_cls=y_pred.round()
    acc=y_pred.eq(y_test).sum()/float(y_test.shape[0])
    print(f"accuracy:{acc}")
