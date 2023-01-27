#Copyright 2021, Jason Lequyer and Laurence Pelletier, All rights reserved.
#Sinai Health SystemLunenfeld-Tanenbaum Research Institute
#600 University Avenue, Room 1070
#Toronto, ON, M5G 1X5, Canada

### 1D jpg


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
#from tifffile import imread, imwrite
import numpy as np
from pathlib import Path
import cv2
import sys
import torch.utils.data as utils_data
import time

class AugmentNoise(object):
    def __init__(self, noisier_noise):
        print('noisier_noise', noisier_noise)
        self.noisier_noise = noisier_noise

    def add_noisier_noise(self, x):
        shape = x.shape
        std = self.noisier_noise / 255.0
        std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
        noise = torch.cuda.FloatTensor(shape, device=x.device)
        torch.normal(mean=0.0,
                     std=std,
                     generator=get_generator(),
                     out=noise)
        return x + noise

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

if __name__ == "__main__":
    operation_seed_counter = 0

    tsince = 100
    folder = sys.argv[1]
    #outfolder = folder+'_N2F'
    outfolder = sys.argv[2]
    noisier_noise = float(sys.argv[3])
    alpha = float(sys.argv[4])

    os.makedirs(outfolder, exist_ok=True)
        
    noise_adder = AugmentNoise(noisier_noise = noisier_noise)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (torch.cuda.is_available()):
        print('gpu')
    else:
        print('cpu')

    class TwoCon(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            return x
    
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, 64)
            self.conv2 = TwoCon(64, 64)
            self.conv3 = TwoCon(64, 64)
            self.conv4 = TwoCon(64, 64)  
            self.conv6 = nn.Conv2d(64,1,1)
            
    
        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x = self.conv4(x3)
            x = torch.sigmoid(self.conv6(x))
            return x
        
    file_list = [f for f in os.listdir(folder)] # argumented folder name
    file_list.sort()
    start_time = time.time()
    for v in range(len(file_list)):
        
        file_name =  file_list[v]
        print(file_name)
        if file_name[0] == '.':
            continue
        
        img = cv2.imread(folder + '/' + file_name, cv2.IMREAD_UNCHANGED)
        typer = type(img[0,0])
        
        minner = np.amin(img)
        img = img - minner
        maxer = np.amax(img)
        img = img/maxer
        img = img.astype(np.float32)
        shape = img.shape
        
        
    
        listimgH = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin.shape[0]):
            for j in range(imgin.shape[1]):
                if j % 2 == 0:
                    imgin[i,j] = imgZ[2*i+1,j]
                    imgin2[i,j] = imgZ[2*i,j]
                if j % 2 == 1:
                    imgin[i,j] = imgZ[2*i,j]
                    imgin2[i,j] = imgZ[2*i+1,j]
        imgin = torch.from_numpy(imgin)
        imgin = torch.unsqueeze(imgin,0)
        imgin = torch.unsqueeze(imgin,0)
        imgin = imgin.to(device)
        imgin2 = torch.from_numpy(imgin2)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = imgin2.to(device)
        listimgH.append(imgin)
        listimgH.append(imgin2)
        
        listimgV = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin3 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin3.shape[0]):
            for j in range(imgin3.shape[1]):
                if i % 2 == 0:
                    imgin3[i,j] = imgZ[i,2*j+1]
                    imgin4[i,j] = imgZ[i, 2*j]
                if i % 2 == 1:
                    imgin3[i,j] = imgZ[i,2*j]
                    imgin4[i,j] = imgZ[i,2*j+1]
        imgin3 = torch.from_numpy(imgin3)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = imgin3.to(device)
        imgin4 = torch.from_numpy(imgin4)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = imgin4.to(device)
        listimgV.append(imgin3)
        listimgV.append(imgin4)
        
    
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(device)
        
        listimgV1 = [[listimgV[0],listimgV[1]]]
        listimgV2 = [[listimgV[1],listimgV[0]]]
        listimgH1 = [[listimgH[1],listimgH[0]]]
        listimgH2 = [[listimgH[0],listimgH[1]]]
        listimg = listimgH1+listimgH2+listimgV1+listimgV2
        
        net = Net()
        net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
        noisier_criterion = torch.nn.MSELoss()
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*105
        last10psnr = [0]*105
        cleaned = 0
        while timesince <= tsince:
            indx = np.random.randint(0,len(listimg))
            data = listimg[indx]
            inputs = data[0]
            labello = data[1]
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss1 = criterion(outputs, labello)

            inputs_noisier = noise_adder.add_noisier_noise(inputs)
            labello_noisier = noise_adder.add_noisier_noise(labello)

            inputs_output = net(inputs_noisier)
            labello_output = net(labello_noisier)
            loss2 = noisier_criterion(inputs_output, inputs) + noisier_criterion(labello_output, labello)


            loss = loss1 + alpha * loss2
            running_loss1+=loss1.item()
            running_loss2+=loss2.item()
            loss.backward()
            optimizer.step()
            
            
            running_loss1=0.0
            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned*maxer+minner)
                outputstest = net(img)
                cleaned = outputstest[0,0,:,:].cpu().detach().numpy()
                
                noisy = img.cpu().detach().numpy()
                ps = -np.mean((noisy-cleaned)**2)
                last10psnr.pop(0)
                last10psnr.append(ps)
                if ps > maxpsnr:
                    maxpsnr = ps
                    outclean = cleaned*maxer+minner
                    timesince = 0
                else:
                    timesince+=1.0
                        
        
        H = np.mean(last10, axis=0)
        
        cv2.imwrite(outfolder + '/' + file_name, H.astype(typer))
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        
        
        torch.cuda.empty_cache()
    

