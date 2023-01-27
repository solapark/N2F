#from tifffile import imread
import numpy as np
import cv2
import os
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


if __name__ == "__main__":
    
    noisydir = sys.argv[1]
    GTdir = sys.argv[2]    
    file_list = [f for f in os.listdir(noisydir)]
    numberofpoints = len(file_list)
    channel = 0    

    counter = 0
    avgps = 0
    avgss = 0
    for v in range(numberofpoints):
        filename = file_list[v]
        filename_GT = filename.replace('real', 'mean')
        if filename[0] == '.':
            continue

        counter += 1
        img = cv2.imread(noisydir + '/' + filename, cv2.IMREAD_UNCHANGED)
        GT = cv2.imread(GTdir + '/' + filename_GT, cv2.IMREAD_UNCHANGED)
        ps = psnr(GT,img,data_range = 255) 
        #ss = ssim(GT, img, gaussian_weights = True, sigma=1.5, use_sample_covariance=False, data_range = 255) #1D image  
        ####ps = psnr(GT,img,data_range = 255, channel_axis=2) #3D image 
        ss = ssim(GT, img, gaussian_weights = True, sigma=1.5, use_sample_covariance=False, data_range = 255, channel_axis=2) #3D image
        avgps += ps
        avgss += ss
    avgps = avgps/counter
    avgss = avgss/counter
    print('PSNR: '+str(avgps))
    print('SSIM: '+str(avgss))
    print(str(avgps), '\t', str(avgss))
