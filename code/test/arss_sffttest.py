import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def phase_generation_sfft(u_in, slm_size, wavelength, prop_dist, dtype=np.complex64):
    field_resolution = u_in.shape
    num_y, num_x = field_resolution[2], field_resolution[3]
    dy, dx = slm_size
    z1 = prop_dist
    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    y = m * dy
    x = n * dx

    X, Y = np.meshgrid(x, y)
    dv = wavelength * z1 / num_y / dy
    du = wavelength * z1 / num_x / dx

    u = m * abs(du)
    v = n * abs(dv)
    U, V = np.meshgrid(u, v)

    phasev = np.exp(1j * np.pi / wavelength / z1 * (U ** 2 + V ** 2))
    phasev = np.array(phasev, dtype=dtype)

    phaseh2 = np.exp(1j * np.pi / wavelength / z1 * (X ** 2 + Y ** 2))
    phaseh2 = np.array(phaseh2, dtype=dtype)

    return phasev, phaseh2

def propagation_ARSS_sfft(u_in, phaseh, phaseu, phasec, phasev, phaseh2, dtype=np.complex64):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    通过使用FFT传播输入场u_in通过传递函数TF。
    """
   

    u = u_in * phaseu
    # u代表了输入场

    U1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(u), axes=(-2, -1), norm='ortho'))

    Trans = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(phaseh), axes=(-2, -1), norm='ortho'))

    U2 = Trans * U1

    u1 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(U2), axes=(-2, -1), norm='ortho'))
    # 下面是我需要修改的
    u_out = u1 * phasec

    u2 = u_out * phasev

    u_out = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(u2), axes=(-2, -1), norm='ortho'))
    u_out = u_out * phaseh2
    return u_out

def phase_generation_Arss(u_in, feature_size, wavelength, prop_dist, dtype=np.complex64, arss_s=1):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    feature_size: (dy, dx) size of the feature in the object plane, which is caculated by dv and arss_s
    prop_dist: z1*totals
    """
    field_resolution = u_in.shape
    num_y, num_x = field_resolution[2], field_resolution[3]
    dy, dx = feature_size
    z1 = prop_dist
    # 这个s也是arss的s，平衡zoom的feature_sizeom
    s = 1 / arss_s

    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    y = m * dy
    x = n * dx

    X, Y = np.meshgrid(x, y)

    # phaseh
    phaseh = np.exp(1j * np.pi / (wavelength * z1) * s * (X ** 2 + Y ** 2))
    phaseh = np.array(phaseh, dtype=dtype)
    
    # phaseu
    phaseu = np.exp(1j * np.pi / (wavelength * z1) * (s ** 2 - s) * (X ** 2 + Y ** 2))
    phaseu = np.array(phaseu, dtype=dtype)

    # phasec
    phasec = np.exp(1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / (1j)
    phasec = np.array(phasec, dtype=dtype)

    return phaseh, phaseu, phasec

if __name__ == "__main__":
    # Model Parameter
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    image_res = (1080, 1080)
    z1=0.1
    arss_s=3.2
    wavelength = 532 * nm
    slm_size=   (8 * um, 8 * um)
    slm_pitch = 8 * um

    #totals=abs(totals)
    totals=2.5
    z2=z1*4
    feature_size = (totals * 8 * um, totals * 8 * um)