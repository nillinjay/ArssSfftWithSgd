import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from scipy.fftpack import *
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os
import time
from fullarss_with_comAmp import fftshift,ifftshift

def propagation_ARSS_sfft(u_in, phaseh, phaseu, phasec,phasev,phaseh2 ,dtype=torch.complex64):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    通过使用FFT传播输入场u_in通过传递函数TF。
    """
    u = u_in * phaseh2
    u =ifftshift(torch.fft.ifftn(ifftshift(u), dim=(-2, -1), norm='ortho'))
    u = u * phasev

    u = u * phaseu
    # u代表了输入场

    U1 = fftshift(torch.fft.fftn(fftshift(u), dim=(-2, -1), norm='ortho'))

    Trans = fftshift(torch.fft.fftn(fftshift(phaseh), dim=(-2, -1), norm='ortho'))

    U2 = Trans * U1

    u1 = ifftshift(torch.fft.ifftn(ifftshift(U2), dim=(-2, -1), norm='ortho'))
    #下面 是我需要修改的
    u_out = u1 * phasec

    u2=u_out*phasev

    u_out=fftshift(torch.fft.fftn(fftshift(u2),dim=(-2,-1),norm='ortho'))
    u_out= u_out*phaseh2
    return u_out


def phase_generation_Arss(u_in, feature_size, wavelength, prop_dist, dtype=torch.complex64,arss_s=1):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    feature_size: (dy, dx) size of the feature in the object plane, which is caculated by dv and arss_s
    prop_dist: z1*totals

    """
    field_resolution = u_in.size()
    num_y, num_x = field_resolution[2], field_resolution[3]
    dy, dx = feature_size
    z1=prop_dist
    #这个s也是arss的s，平衡zoom的feature_sizeom
    s=1/arss_s

    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    y = m * dy
    x = n * dx

    X, Y = np.meshgrid(x, y)

    # phaseh
    phaseh = np.exp(1j * np.pi / (wavelength * z1) * s * (X ** 2 + Y ** 2))
    phaseh = phaseh.reshape(1, 1, phaseh.shape[0], phaseh.shape[1])
    phaseh = torch.tensor(phaseh, dtype=dtype).to(u_in.device)
    
    # phaseu
    phaseu = np.exp(1j * np.pi / (wavelength * z1) * (s ** 2 - s) * (X ** 2 + Y ** 2))
    phaseu = phaseu.reshape(1, 1, phaseu.shape[0], phaseu.shape[1])
    phaseu = torch.tensor(phaseu, dtype=dtype).to(u_in.device)

    # phasec
    # phasec = np.exp(1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / (1j * wavelength * prop_dist)
    phasec = np.exp(1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / (1j)
    phasec = phasec.reshape(1, 1, phasec.shape[0], phasec.shape[1])
    phasec = torch.tensor(phasec, dtype=dtype).to(u_in.device)

    return phaseh, phaseu,phasec

def phase_generation_sfft(u_in,slm_size,wavelength,prop_dist,dtype=torch.complex64):
    field_resolution = u_in.size()
    num_y, num_x = field_resolution[2], field_resolution[3]
    dy, dx = slm_size
    z1=prop_dist
    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    y = m * dy
    x = n * dx


    X, Y = np.meshgrid(x, y)
    dv=wavelength*z1/num_y/dy
    du=wavelength*z1/num_x/dx

    u=m*abs(du)
    v=n*abs(dv)
    U,V=np.meshgrid(u,v)

    phasev=np.exp(1j*np.pi/wavelength/z1*(U**2+V**2))
    phasev=phasev.reshape(1,1,phasev.shape[0],phasev.shape[1])
    phasev=torch.tensor(phasev,dtype=dtype).to(u_in.device)

    phaseh2=np.exp(1j*np.pi/wavelength/z1*(X**2+Y**2))
    phaseh2=phaseh2.reshape(1,1,phaseh2.shape[0],phaseh2.shape[1])
    phaseh2=torch.tensor(phaseh2,dtype=dtype).to(u_in.device)

    return phasev,phaseh2

def cac_dv(N,slm_size,wavelength,prop_dist,dtype=torch.complex64):
    
    dy, dx = slm_size
    z1=prop_dist
    dv=wavelength*z1/dy/N
    #这个s也是arss的s，平衡zoom的feature_sizeom

    return dv
def cac_totals(s,dv,slmres):
    totals=s*dv/slmres
    return totals

def clight_generation(u_in, wavelength,totals,z2):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    """
    field_resolution = u_in.shape
    num_y, num_x = field_resolution[1], field_resolution[0]

    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    # c_light
    s = totals  # 缩放参数

    dx0 = 8e-6
    dy0 = 8e-6

    xm0 = dx0 * m
    ym0 = dy0 * n
    xx0, yy0 = np.meshgrid(xm0, ym0)


    # 收敛光
    c_x = 1  # 收敛光收敛角度调整
    c_y = 1
    c_light = np.exp(1j * np.pi * (s ** 2 / (wavelength * (z2 * c_x)) * xx0 ** 2 + s ** 2 / (wavelength * (z2 * c_y)) * yy0 ** 2))
    return c_light
    


if __name__=="__main__":
    mm,um,nm=1e-3,1e-6,1e-9
    image_res = (1080, 1080)
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *image_res))
    pad = torch.nn.ZeroPad2d((1080 // 2, 1080 // 2, 1080 // 2, 1080 // 2))
    field = pad(init_phase)
    slm_size = (8*um, 8*um)
    feature_size = (15*um, 15*um)
    wavelength = 500*nm
    z1 = 1
    z2=0.1
    s=10
    phaseh, phaseu,phasec = phase_generation_Arss(u_in=field, feature_size=feature_size, wavelength=wavelength, prop_dist=z1)
    phasev,phaseh2=phase_generation_sfft(u_in=field,slm_size=slm_size,wavelength=wavelength,prop_dist=z2)
    u_out = propagation_ARSS_sfft(field, phaseh, phaseu,phasec,phasev,phaseh2)
    dv=cac_dv(1080,slm_size,wavelength,z2)
    totals=cac_totals(s,dv,8*um)
    clight=clight_generation(field,wavelength,totals,z1)
    print(u_out.shape)