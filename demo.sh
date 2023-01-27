[dataset]
/data3/sap/dataset
1. BSD68
clean : BSD68_jpg
noisy25 : BSD68_jpg_gaussian25
noisy50 : BSD68_jpg_gaussian50

2. PolyU
clean : PolyU_crop_mean
noisy : PolyU_crop_real

3. Confocal
clean : Confocal (tif)
noisy : Confocal_gaussianpoisson

4. Set12
clean : Set12 (tif)

5. livecells
clean : livecells (tif)

[output]
1D N2F : N2F_jpg.py 
3D N2F: N2F_3D_jpg.py
1D Ne2Ne : Ne2Ne_jpg.py
compute : compute_psnr_ssim_jpg.py 

[cmd]
# N2F
-bsd68
python N2F_jpg.py /data3/sap/dataset/BSD68_jpg_gaussian25 /data3/sap/N2F/N2F/BSD68_gaussian25

# N2F_noisier2noise a=1
-bsd68
python N2F_noisier2noise.py /data3/sap/dataset/BSD68_jpg_gaussian25 /data3/sap/N2F/N2F_nr2n/BSD68_gaussian25 25

# N2F_noisier2noise a=.5
-bsd68
python N2F_noisier2noise.py /data3/sap/dataset/BSD68_jpg_gaussian25 /data3/sap/N2F/N2F_nr2n_a.5/BSD68_gaussian25 25 .5

# N2F_3D
-polyU
python N2F_3D_jpg.py /data3/sap/dataset/PolyU_crop_real /data3/sap/N2F/N2F/PolyU_crop_real

# N2F_noisier2noise a=1
python N2F_3D_noisier2noise.py /data3/sap/dataset/PolyU_crop_real /data3/sap/N2F/N2F_nr2n/PolyU_crop_real 1

# N2F_noisier2noise a=.5
python N2F_3D_noisier2noise.py /data3/sap/dataset/PolyU_crop_real /data3/sap/N2F/N2F_nr2n_a.5/PolyU_crop_real .5


# Ne2Ne
python Ne2Ne_jpg.py /hdd_4T/jes/Noise2Fast/BSD68_jpg_gaussian50 BSD68_jpg_gaussian50_Ne2Ne

# compute psnr, ssim
-bsd68
python compute_psnr_ssim_jpg.py /data3/sap/N2F/N2F/BSD68_gaussian25 /data3/sap/dataset/BSD68_jpg 

-bsd68_nr2n
python compute_psnr_ssim_jpg.py /data3/sap/N2F/N2F_nr2n/BSD68_gaussian25 /data3/sap/dataset/BSD68_jpg 

-bsd68_nr2n alpha=.5
python compute_psnr_ssim_jpg.py /data3/sap/N2F/N2F_nr2n_a.5/BSD68_gaussian25 /data3/sap/dataset/BSD68_jpg 

-polyU
python compute_psnr_ssim_3D_jpg.py /data3/sap/N2F/N2F/PolyU_crop_real /data3/sap/dataset/PolyU_crop_mean


# synth noise
python add_gaussian_noise_jpg.py /hdd_4T/jes/Noise2Fast/BSD68_jpg 25
--> python add_gaussian_noise_jpg.py (data path) (std)


