"""
Author: Jie Zhou
Affiliation: Sichuan University
Date: August 7, 2024
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

import time


def create_circle(radius, image_size=1080):
    """
    创建一个半径为radius的圆形。

    参数:
    radius -- 圆的半径
    image_size -- 图像的尺寸，格式为(width, height)

    返回:
    一个表示圆形的二维张量，其中圆内部为1，外部为0。
    """
    # 创建一个空的图像张量，初始化为0
    y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size))

    # 计算每个点到圆心的距离
    center_x, center_y = image_size // 2, image_size // 2
    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # 创建一个圆形，圆内的值为1，圆外的值为0
    circle = (distance <= radius).float()

    return circle

def double_phase(Uf):
    # Uf shape: [batch_size, channels, height (N), width (M)]
    N, M = Uf.shape[-2], Uf.shape[-1]  # Extract height and width

    # Generate coordinate grids
    x = torch.arange(M, device=Uf.device).reshape(1, M).expand(N, M)
    y = torch.arange(N, device=Uf.device).reshape(N, 1).expand(N, M)

    # Create Mask1 using cosine squared
    Mask1 = torch.cos(np.pi * (x + y) / 2).pow(2)
    Mask2 = 1 - Mask1  # Inverse of Mask1

    # Remove batch and channel dimensions for computation
    Uf = Uf.squeeze(0).squeeze(0)  # Now Uf has shape [N, M]

    # Compute amplitude and phase
    Uf_P = torch.angle(Uf)
    Uf_A = torch.abs(Uf)
    w = Uf_A / torch.max(Uf_A)

    # Compute theta1 and theta2
    theta1 = Uf_P + torch.acos(w)
    theta2 = Uf_P - torch.acos(w)

    # Combine phases using the masks
    theta = theta1 * Mask1 + theta2 * Mask2

    # Add batch and channel dimensions back
    theta = theta.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N, M]

    return theta


def pading(U):
    """
    Pad the input image U to a shape of (1080, 1920).
    """
    m, n = U.shape
    pad = np.zeros((1080, 1080), dtype=np.complex64)

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
    shifts the width and heights
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


def propagation_ARSS(u_in, phaseh, phaseu, phasec, dtype=torch.complex64):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    """
    u = u_in * phaseu

    U1 = fftshift(torch.fft.fftn(fftshift(u), dim=(-2, -1), norm='ortho'))

    Trans = fftshift(torch.fft.fftn(fftshift(phaseh), dim=(-2, -1), norm='ortho'))

    U2 = Trans * U1

    u1 = ifftshift(torch.fft.ifftn(ifftshift(U2), dim=(-2, -1), norm='ortho'))

    u_out = u1 * phasec

    # print(u_out.size()[2], u_out.size()[3]) 为什么尺寸会变。。。

    return u_out


def clight_generation(u_in, wavelength, d):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    """
    field_resolution = u_in.shape
    num_y, num_x = field_resolution[1], field_resolution[0]

    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    # c_light
    s = 1  # 缩放参数

    dx0 = 20e-6
    dy0 = 20e-6

    xm0 = dx0 * m
    ym0 = dy0 * n
    xx0, yy0 = np.meshgrid(xm0, ym0)

    z1 = d
    # 收敛光
    c_x = 1  # 收敛光收敛角度调整
    c_y = 1
    c_light = np.exp(
        -1j * np.pi * (s ** 2 / (wavelength * (z1 * c_x)) * xx0 ** 2 + s ** 2 / (wavelength * (z1 * c_y)) * yy0 ** 2))
    return c_light


def phase_generation(u_in, feature_size, origin_size, wavelength, prop_dist, dist, model, dtype=torch.complex64):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    """
    field_resolution = u_in.size()
    num_y, num_x = field_resolution[2], field_resolution[3]
    dy_h, dx_h = feature_size
    dy_o, dx_o = origin_size

    dy = (wavelength * dist) / (num_y * dy_h)
    dx = (wavelength * dist) / (num_x * dx_h)

    if model == 1:
        s = dy_o / dy
    else:
        s = dy / dy_o

    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    y = m * dy
    x = n * dx

    X, Y = np.meshgrid(x, y)

    # phaseh
    phaseh = np.exp(1j * np.pi / (wavelength * prop_dist) * s * (X ** 2 + Y ** 2))
    phaseh = phaseh.reshape(1, 1, phaseh.shape[0], phaseh.shape[1])
    phaseh = torch.tensor(phaseh, dtype=dtype).to(u_in.device)

    # phaseu
    phaseu = np.exp(1j * np.pi / (wavelength * prop_dist) * (s ** 2 - s) * (X ** 2 + Y ** 2))
    phaseu = phaseu.reshape(1, 1, phaseu.shape[0], phaseu.shape[1])
    phaseu = torch.tensor(phaseu, dtype=dtype).to(u_in.device)

    # phasec
    phasec = np.exp(1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / (1j * wavelength * prop_dist)
    # phasec = np.exp(
    #     1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / (1j)
    phasec = phasec.reshape(1, 1, phasec.shape[0], phasec.shape[1])
    phasec = torch.tensor(phasec, dtype=dtype).to(u_in.device)

    return phaseh, phaseu, phasec


def fresnel_diffraction(u_in, feature_size, wavelength, dist):
    field_resolution = u_in.size()
    num_y, num_x = field_resolution[2], field_resolution[3]
    if dist > 0:
        dy_h, dx_h = feature_size
        dy_v = (wavelength * dist) / (num_y * dy_h)
        dx_v = (wavelength * dist) / (num_x * dx_h)
    else:
        dy_v, dx_v = feature_size
        dy_h = (wavelength * -dist) / (num_y * dy_v)
        dx_h = (wavelength * -dist) / (num_x * dx_v)
    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    y_v = m * dy_v
    x_v = n * dx_v

    X_v, Y_v = np.meshgrid(x_v, y_v)

    y_h = m * dy_h
    x_h = n * dx_h

    X_h, Y_h = np.meshgrid(x_h, y_h)

    trans = np.exp(1j * np.pi * (X_v ** 2 + Y_v ** 2) / (wavelength * dist))
    trans = trans.reshape(1, 1, trans.shape[0], trans.shape[1])
    trans = torch.tensor(trans, dtype=torch.complex64).to(u_in.device)

    C = np.exp(1j * np.pi * (X_h ** 2 + Y_h ** 2) / (wavelength * dist))
    C = C.reshape(1, 1, C.shape[0], C.shape[1])
    C = torch.tensor(C, dtype=torch.complex64).to(u_in.device)

    u = u_in * trans

    u_f = fftshift(torch.fft.fftn(fftshift(u), dim=(-2, -1), norm='ortho'))

    u_h = C * u_f

    return u_h


if __name__ == "__main__":
    # Model Parameter
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    prop_dist = -0.4
    dist_2 = -0.1
    wavelength = 532 * nm
    feature_size = (8 * um, 8 * um)
    origin_size = (20 * um, 20 * um)
    slm_pitch = 8 * um
    image_res = (1080, 1080)

    # Training Parameter
    dtype = torch.float32
    device = torch.device('cuda')
    loss = nn.MSELoss().to(device)
    propagator = propagation_ARSS
    diff_2 = fresnel_diffraction

    # Image Processing
    target = cv2.imread('Lena.png')
    target = cv2.resize(target, (1080, 1080))
    target_amp = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    c_light = clight_generation(u_in=target_amp, wavelength=wavelength, d=-prop_dist)
    target_camp = target_amp * c_light / 255
    target_camp = pading(target_camp)
    target_camp = np.array(target_camp)
    target_camp = target_camp.reshape(1, 1, target_camp.shape[0], target_camp.shape[1])
    target_camp = torch.tensor(target_camp, dtype=torch.complex64).to(device)
    # plt.title('complex_hologram')
    # plt.imshow(target_camp.angle().squeeze().cpu().detach().numpy(), cmap='gray')
    # plt.show()

    # complex_hologram
    phaseh, phaseu, phasec = phase_generation(u_in=target_camp, feature_size=(8 * um, 8 * um),
                                              origin_size=(20 * um, 20 * um), wavelength=wavelength,
                                              prop_dist=-prop_dist, dist=-dist_2, model=1)
    complex_hologram = propagator(u_in=target_camp, phaseh=phaseh, phaseu=phaseu, phasec=phasec)

    complex_hologram_1 = diff_2(u_in=complex_hologram, feature_size=feature_size, wavelength=wavelength, dist=-dist_2)

    double_phase_hologram = torch.exp(1j * double_phase(complex_hologram_1))
    # plt.title('complex_hologram')
    # plt.imshow(complex_hologram_1.angle().squeeze().cpu().detach().numpy(), cmap='gray')
    # plt.show()

    # recon

    complex_hologram_2 = diff_2(u_in=complex_hologram_1, feature_size=feature_size, wavelength=wavelength, dist=dist_2)

    # plt.title('complex_hologram')
    # plt.imshow(complex_hologram_2.abs().squeeze().cpu().detach().numpy(), cmap='gray')
    # plt.show()
    # filter_c = create_circle(1.65 * mm)
    # complex_hologram = complex_hologram * filter_c

    phaseh, phaseu, phasec = phase_generation(u_in=complex_hologram_2, feature_size=(8 * um, 8 * um),
                                              origin_size=(20 * um, 20 * um), wavelength=wavelength,
                                              prop_dist=prop_dist, dist=-dist_2, model=0)
    recon_field = propagator(u_in=complex_hologram_2, phaseh=phaseh, phaseu=phaseu, phasec=phasec)

    recon_amp = recon_field.angle()
    recon_amp = np.array(recon_amp.data.cpu()[0])[0]
    plt.title('Reconstruction')
    plt.imshow(recon_field.abs().squeeze().cpu().detach().numpy(), cmap='gray')
    plt.show()

