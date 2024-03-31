#%%
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import cv2 as cv
import os

#%%
#数据集获取
# 定义数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载训练数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

# 加载测试数据集
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

#%%
#搭建SVM模型
class SVM(nn.Module):
    def __init__(self,classification):
        super(SVM,self).__init__()
        self.fc = nn.Linear(28*28,classification)
    def forward(self,x):
        x = x.view(x.size(0), -1)  # 将图像数据展平
        x = self.fc(x)
        return x

#%%Hinge损失函数
def Hinge_loss():
    return nn.HingeEmbeddingLoss()

    #%%
#训练
def train_svm(model, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = nn.functional.one_hot(labels, num_classes=10)
            labels = labels*2-1
            #print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

#%%
#模型评价
def test_svm(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {100 * correct / total}%")

#%%
#开始训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
svm_model = SVM(10).to(device)
criterion = Hinge_loss()
optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)

# 训练模型
train_svm(svm_model, trainloader, criterion, optimizer, epochs=10)

# 测试模型
test_svm(svm_model, testloader)
#保存模型
torch.save(svm_model,"svm_model.pth")


#%%
svm_model = torch.load("svm_model.pth")
img_pth = './picture/'
file = os.listdir(img_pth)

for pic in file:
    img =  cv.imread(img_pth + pic)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if img.shape == (28,28):
        torch.no_grad()
        #img = img.squeeze(0)
        img = torch
        res = svm_model.forward(img)
        print(res)