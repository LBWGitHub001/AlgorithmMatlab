#%%模型部署
import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from torch import nn
import os

#%%
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # 10类输出

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将图像数据展平
        return self.fc(x)

#%%读取模型
net = torch.load('svm_model.pth')
net.eval()

#%%读取图片
img_pth = './picture/'
file = os.listdir(img_pth)
res_pth = './result/'

for pic in file:
    img =  cv.imread(img_pth + pic)
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if image.shape == (28,28):
        torch.no_grad()
        #img = img.squeeze(0)
        traf = transforms.ToTensor()
        img = traf(image)
        img = torch.unsqueeze(img,dim=0)
        img = img.cuda()
        res = net(img)
        res = torch.argmax(res).tolist()
        res = str(res)
        print(pic+' is '+res)
        cv.imwrite(res_pth+pic[0:-4]+'detect'+res+'.png',image)

