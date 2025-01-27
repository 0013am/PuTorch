import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device config
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameter
input_size= 784   # 28*28
hidden_size=100
num_classes =10
num_epochs=2
batch_size=100
learning_rate=0.01

#MNIST
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


for batch_idx, (data, target) in enumerate(train_loader):
    print(batch_idx,data.shape,target.shape)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(data[i][0],cmap='gray')
    plt.show
    break

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetwork,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return out

model=NeuralNetwork(input_size,hidden_size,num_classes)
#loss and optim
criterian=nn.CrossEntropyLoss()
optmizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#training
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_loader):
        #100,1,28,28 -> 100,784
        images=images.reshape(-1,784).to(device)
        labels=labels.to(device)
        # forward pass
        outputs=model(images)
        loss=criterian(outputs,labels)
        #backward
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

        if (i+1)%10==0:
            print(f'epoch:{epoch+1}/ {num_epochs},step {i+1}{n_total_steps},loss={loss.item():.4f}')

# testing
with torch.no_grad():
    n_correct=0
    n_samples=0
    for img,lab in test_loader:
        img=img.reshape(-1,784).to(device)
        lab=lab.to(device)
        out=model(img)

        # value,index
        _,pred=torch.max(out,1)
        n_samples+=lab.shape[0]
        n_correct+=(pred==lab).sum().item()
acc=100.0*n_correct/n_samples
print(f'accuracy ={acc}')



