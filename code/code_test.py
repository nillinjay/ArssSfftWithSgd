"""
Author: nillin jay
reference: Zhou jie
Affiliation: Sichuan University
Date: August 9, 2024 (Modified)
Description: Quality improvement of unfiltered holography by optimizing high diffraction orders with fill factor
"""
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
from method import *

def pading(U):
    """
    Pad the input image U to a shape of (1080, 1080).
    """
    m, n = U.shape
    pad = np.zeros((1080, 1080),dtype=np.complex64)

    pad[1080 // 2 - m // 2:1080 // 2 + m // 2, 1080 // 2 - n // 2:1080 // 2 + n // 2] = U
    return pad

def polar_to_rect(mag, ang):
    """
    Convert polar coordinates to rectangular coordinates.
    """
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def ifftshift(tensor):
    """
    ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]
    shifts the width and heights+
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted

def fftshift(tensor):
    """
    fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]
    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted

def roll_torch(tensor, shift, axis):
    """
    implements numpy roll() or Matlab circshift() functions for tensors
    """
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)

    return torch.cat([after, before], axis)






class SGD(nn.Module):
    def __init__(self, phaseh, phaseu, phasec,phasev,phaseh2, feature_size, wavelength, prop_dist, num_iters, propagator=None,
                 loss=nn.MSELoss(), lr=0.1, lr_s=0.003, s0=1.0, device=torch.device('cuda')):
        """
        Initialize the SGD optimization model.
        """
        super(SGD, self).__init__()
        # Setting parameters
        self.phaseh = phaseh
        self.phaseu = phaseu
        self.phasec = phasec
        self.phasev=phasev
        self.phaseh2=phaseh2
        self.feature_size = feature_size
        self.wavelength = wavelength
        self.prop_dist = prop_dist
        self.prop = propagator
        self.num_iters = num_iters
        self.lr = lr
        self.lr_s = lr_s
        self.init_scale = s0
        self.dev = device
        self.loss = loss.to(device)

    def forward(self, target_amp, init_phase=None):
        """
        Perform forward pass of SGD optimization.
        """
        final_phase = stochastic_gradient_descent(init_phase, target_amp, self.phaseh, self.phaseu, self.phasec,self.phasev,self.phaseh2, self.num_iters,
                                                  self.feature_size, self.wavelength, self.prop_dist, propagator=self.prop,
                                                  loss=self.loss, lr=self.lr, lr_s=self.lr_s,
                                                  s0=self.init_scale)
        return final_phase


def stochastic_gradient_descent(init_phase, target_amp, phaseh, phaseu, phasec,phasev,phaseh2, num_iters, feature_size, wavelength, prop_dist, propagator=None,
                                loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1, dtype=torch.float32):
    """
    Perform stochastic gradient descent to optimize the phase.
    """
    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)
    slm_phase = init_phase.requires_grad_(True)
    optvars = [{'params': slm_phase}]
    if lr_s > 0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    for k in range(num_iters):
        optimizer.zero_grad()
        real, imag = polar_to_rect(torch.ones_like(slm_phase), slm_phase)#全相位编码
        slm_field = torch.complex(real, imag)

        pad = torch.nn.ZeroPad2d((1080 // 2, 1080 // 2, 1080 // 2, 1080 // 2))
        slm_field_pad = pad(slm_field)

        recon_field = propagator(u_in=slm_field_pad, phaseh=phaseh, phaseu=phaseu, phasec=phasec,phasev=phasev,phaseh2=phaseh2)

        # recon_amp = recon_field.abs()

        recon_field = recon_field[:, :,
                     slm_field_pad.size()[2] // 2 - slm_field.size()[2] // 2:slm_field_pad.size()[2] // 2 +
                                                                             slm_field.size()[2] // 2, \
                     slm_field_pad.size()[3] // 2 - slm_field.size()[3] // 2:slm_field_pad.size()[3] // 2 +
                                                                             slm_field.size()[3] // 2]

        # recon_amp2 = recon_amp1[:, :, slm_field.size()[2] // 2 - 960 // 2:slm_field.size()[2] // 2 + 960 // 2, \
        #              slm_field.size()[3] // 2 - 1680 // 2:slm_field.size()[3] // 2 + 1680 // 2]

        # target_amp1 = target_amp[:, :, slm_field.size()[2] // 2 - 960 // 2:slm_field.size()[2] // 2 + 960 // 2, \
        #               slm_field.size()[3] // 2 - 1680 // 2:slm_field.size()[3] // 2 + 1680 // 2]

        recon_real, recon_imag = polar_to_rect(recon_field.abs(), recon_field.angle())
        recon_amp = recon_field.abs()
        tar_real, tar_imag = polar_to_rect(target_amp.abs(), target_amp.angle())
        tar_amp = target_amp.abs()

        lossValue = loss(s * recon_real, tar_real) + loss(s * recon_imag, tar_imag) + 2 * loss(s * recon_amp, tar_amp)
        print(s, lossValue)
        lossValue.backward()
        optimizer.step()
        with torch.no_grad():
            if k % 500 == 0:

                recon = np.array(recon_amp.data.cpu()[0])[0]
                target = np.array(tar_amp.data.cpu()[0])[0]

                recon = recon / recon.max()
                target = target / target.max()

                PSNR = psnr(recon, target)
                SSIM = ssim(recon, target, data_range=target.max() - target.min())

                print("iteration:{}".format(k))
                print("PSNR:", PSNR)
                print("SSIM:", SSIM)
                print(lossValue)

                plt.subplot(1, 3, 1)
                plt.title('Target')
                plt.imshow(target, cmap='gray')
                plt.subplot(1, 3, 2)
                plt.title('Holo')
                plt.imshow(slm_phase.squeeze().cpu().detach().numpy(), cmap='gray')
                plt.subplot(1, 3, 3)
                plt.title('Reconstruction')
                plt.imshow(recon, cmap='gray')
                plt.show()

    return slm_phase


if __name__ == "__main__":
    # Model Parameter
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    image_res = (1080, 1080)
    z1=-0.2
    arss_s=3.2
    wavelength = 532 * nm
    slm_size=   (8 * um, 8 * um)
    slm_pitch = 8 * um
    dv=abs(cac_dv(1080,slm_size,wavelength,z1))
    fft_s=dv/8/um
    #totals=abs(totals)
    totals=2.5
    z2=z1*4
    feature_size = (totals * 8 * um, totals * 8 * um)
 
    image_res = (1080, 1080)
    k = 2 * np.pi / wavelength
    fill_rate = 0.87
    orders = 3
    # padding = math.ceil(prop_dist/slm_pitch * np.tan(np.arcsin(3*wavelength/(2*slm_pitch))))

    # Training Parameter
    dtype = torch.float32
    device = torch.device('cuda')
    loss = nn.MSELoss().to(device)
    propagator = propagation_ARSS_sfft
    num_iters = 2001
    lr = 0.04
    lr_s = 0.01
    s0 = 1.0

    # Image Processing
    target = cv2.imread('../jpg/image2.jpg')
    target = cv2.resize(target, (1080, 1080))
    target_amp = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    c_light = clight_generation(u_in=target_amp, wavelength=wavelength,totals=totals,z2=z2)
    target_camp = target_amp * c_light / 255

    target_camp = pading(target_camp)
    target_camp = np.array(target_camp)
    target_camp = target_camp.reshape(1, 1, target_camp.shape[0], target_camp.shape[1])
    target_camp = torch.tensor(target_camp, dtype=torch.complex64).to(device)
    plt.title('Holo')
    plt.imshow(target_camp.angle().squeeze().cpu().detach().numpy(), cmap='gray')
    plt.show()
    # Initial Phase Pattern
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *image_res)).to(device)

    # phase
    pad = torch.nn.ZeroPad2d((1080 // 2, 1080 // 2, 1080 // 2, 1080 // 2))
    field = pad(init_phase)
    phaseh, phaseu, phasec = phase_generation_Arss(u_in=field, feature_size=feature_size, wavelength=wavelength, prop_dist=z2,arss_s=arss_s)
    phasev,phaseh2 ,rect= phase_generation_sfft(u_in=field, slm_size=slm_size, wavelength=wavelength, prop_dist=z1)
    phaseh=phaseh*rect
    # training staring
    sgd = SGD(phaseh=phaseh, phaseu=phaseu, phasec=phasec,phasev=phasev,phaseh2=phaseh2 ,feature_size=feature_size, wavelength=wavelength, prop_dist=z1+z2,num_iters=num_iters, propagator=propagator, loss=loss, lr=lr, lr_s=lr_s, s0=s0, device=device)
    final_phase = sgd(target_amp=target_camp, init_phase=init_phase)

    # Hologram Preservation
    final_phase = np.array(final_phase.data.cpu()[0])[0]
    final_phase = ((final_phase + np.pi) % (2 * np.pi)) / 2 / np.pi * 255
    output_dir = '../experiment'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'Circle_HOLO_43.png')
    plt.imsave(output_path, final_phase, cmap='gray')

