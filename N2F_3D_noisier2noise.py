#Copyright 2021, Jason Lequyer and Laurence Pelletier, All rights reserved.
#Sinai Health SystemLunenfeld-Tanenbaum Research Institute
#600 University Avenue, Room 1070
#Toronto, ON, M5G 1X5, Canada

### 3D jpg


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


if __name__ == "__main__":
    tsince = 100
    folder = sys.argv[1]
    #####outfolder = folder+'_N2F'
    outfolder = sys.argv[2]
    os.makedirs(outfolder, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(torch.cuda.is_available()):
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
            self.conv1 = TwoCon(3, 64)
            self.conv2 = TwoCon(64, 64)
            self.conv3 = TwoCon(64, 64)
            self.conv4 = TwoCon(64, 64)  
            self.conv6 = nn.Conv2d(64,3,1)
            
    
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
        img = img.astype(np.float32)
        typer = type(img[0,0,0])
        minner = np.amin(img, axis=(0,1))        
        maxer = np.amax(img, axis=(0,1))

        for c in range(3):
            img_per_channel = img[:, :, c]
            img_per_channel = img_per_channel - minner[c]
            img_per_channel = img_per_channel/(maxer[c]+1.0e-20)
            img[:, :, c] = img_per_channel
        
        #img = img.astype(np.float32)
        shape = img.shape
    
        listimgH = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1], :]
        
        imgin = np.zeros((Zshape[0]//2,Zshape[1], 3),dtype=np.float32)
        imgin2 = np.zeros((Zshape[0]//2,Zshape[1], 3),dtype=np.float32)
        for i in range(imgin.shape[0]):
            for j in range(imgin.shape[1]):
                if j % 2 == 0:
                    imgin[i,j, :] = imgZ[2*i+1,j, :]
                    imgin2[i,j, :] = imgZ[2*i,j, :]
                if j % 2 == 1:
                    imgin[i,j, :] = imgZ[2*i,j, :]
                    imgin2[i,j, :] = imgZ[2*i+1,j, :]
        imgin = torch.from_numpy(imgin)
        #imgin = torch.unsqueeze(imgin,0)
        imgin = torch.unsqueeze(imgin,0)
        imgin = imgin.to(device)
        imgin = imgin.permute(0, 3, 1, 2)

        imgin2 = torch.from_numpy(imgin2)
        #imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = imgin2.to(device)
        imgin2 = imgin2.permute(0, 3, 1, 2)

        listimgH.append(imgin)
        listimgH.append(imgin2)
        
        listimgV = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1], :]
        
        imgin3 = np.zeros((Zshape[0],Zshape[1]//2, 3),dtype=np.float32)
        imgin4 = np.zeros((Zshape[0],Zshape[1]//2, 3),dtype=np.float32)
        for i in range(imgin3.shape[0]):
            for j in range(imgin3.shape[1]):
                if i % 2 == 0:
                    imgin3[i,j, :] = imgZ[i,2*j+1, :]
                    imgin4[i,j, :] = imgZ[i, 2*j, :]
                if i % 2 == 1:
                    imgin3[i,j, :] = imgZ[i,2*j, :]
                    imgin4[i,j] = imgZ[i,2*j+1]
        imgin3 = torch.from_numpy(imgin3)
        #imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = imgin3.to(device)
        imgin3 = imgin3.permute(0, 3, 1, 2)

        imgin4 = torch.from_numpy(imgin4)
        #imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = imgin4.to(device)
        imgin4 = imgin4.permute(0, 3, 1, 2)

        listimgV.append(imgin3)
        listimgV.append(imgin4)
        
    
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img,0)
        #img = torch.unsqueeze(img,0)
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
        
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*105
        last10psnr = [0]*105
        cleaned = 0

        count = 0
        #noisy = img.permute(0,2,3,1)
        noisy = img[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
        for c in range(3):
            noisy[:,:,c] = noisy[:,:,c]*maxer[c]+minner[c]
        while timesince <= tsince:
            indx = np.random.randint(0,len(listimg))
            data = listimg[indx]
            inputs = data[0]
            labello = data[1]
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss1 = criterion(outputs, labello)
            loss = loss1
            running_loss1+=loss1.item()
            loss.backward()
            optimizer.step()
            
            
            running_loss1=0.0
            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned)
                outputstest = net(img).permute(0,2,3,1)
                cleaned = outputstest[0,:,:,:].cpu().detach().numpy()
                for c in range(3):
                    cleaned[:,:,c] = cleaned[:,:,c]*maxer[c]+minner[c]
                
                ##noisy = img.permute(0,2,3,1)
                #noisy = img[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
                #for c in range(3):
                #    noisy[:,:,c] = noisy[:,:,c]*maxer[c]+minner[c]
                ps = -np.mean((noisy-cleaned)**2)
                #cv2.imwrite('/home/jes/test/' + str(count) +'_'+file_name, cleaned.astype(typer))
                count = count + 1
                last10psnr.pop(0)
                last10psnr.append(ps)
                if ps > maxpsnr:
                    maxpsnr = ps
                    #outclean = cleaned*maxer+minner
                    timesince = 0
                else:
                    timesince+=1.0
                        
        
        H = np.mean(last10, axis=0)
        
        cv2.imwrite(outfolder + '/' + file_name, H.astype(typer))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('iteration: ' +str(count))
        start_time = time.time()
        
        
        
        torch.cuda.empty_cache()
    
