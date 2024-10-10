"""
double phase encoding ，double step diffraction（没有用常琛亮的两步方法，而是直接ARSS衍射到目标平面，越过收敛光的收敛点）
通过改变全息面到物面的距离使得全息面可以接受全部的物体信息
"""

from numpy import *
from scipy import misc
import tkinter as tk
import tkinter.filedialog
from scipy.fftpack import *
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy import signal
from skimage import measure
import time


def normalized_to_255(u2):
    """
    归一化函数
    :param u2_ang: 输入数据
    :return: 归一化数据
    """
    u2_max = amax(u2)
    u2_min = amin(u2)
    u3_ang = (u2 - u2_min) / (u2_max - u2_min)
    return u3_ang * 255


def ARSS_frenel_function(u, z):
    m = arange(-M1 / 2, M1 / 2)
    n = arange(-N1 / 2, N1 / 2)

    mm, nn = meshgrid(m, n)

    s = 3  # 缩放参数
    t = 3

    ox = 8e-6 * 0  # 平移参数
    oy = 8e-6 * 0

    dx0 = 8e-6
    dy0 = 8e-6

    xm0 = dx0 * m
    ym0 = dy0 * n
    xx0, yy0 = meshgrid(xm0, ym0)

    dx1 = 8e-6
    dy1 = 8e-6

    xm1 = dx1 * m
    ym1 = dy1 * n
    xx1, yy1 = meshgrid(xm1, ym1)

    # 收敛光
    c_x = 1  # 收敛光收敛角度调整
    c_y = 1
    c_light = exp(-j1 * pi * (
                s ** 2 / (h * (z1 * c_x)) * xx0 ** 2 + t ** 2 / (h * (z1 * c_y)) * yy0 ** 2))
    u_patch = zeros((N1, M1), dtype=complex)  # 扩大物平面
    u_patch[int(N1/2 - N/2):int(N1/2 + N/2), int(M1/2 - M/2):int(M1/2 + M/2)] = u

    # 物光乘以收敛光
    u_c = u_patch * c_light
    rect = logical_and(abs(mm) < M/2, abs(nn) < N/2)

    # 常数相位
    C = exp(j1 * h * z + j1 * pi / (h * z) * (
                ((1 - s) * xx1 ** 2 + 2 * ox * xx1 + ox ** 2) + ((1 - t) * yy1 ** 2 + 2 * oy * yy1 + oy ** 2))) / (
                    j1 * h * z) #* rect

    u_re = u_c * exp(j1 * pi / (h*z) * (((s**2 - s)*xx0**2 - 2 * s * ox * xx0) + ((t**2 - t)*yy0**2 - 2 * t * oy * yy0))) #* rect
    uf = fft2(u_re, (N1, M1))

    trans = exp(j1 * pi * ((s * xx1 ** 2) + (t * yy1 ** 2))/(h*z))
    rect2 = logical_and(abs(mm) < M, abs(nn) < N)

    trans = trans #* rect2
    trans = fft2(trans, (N1, M1))
    trans_abs = amax(abs(trans))
    #trans = trans_abs * exp(j1 * angle(trans))

    II4 = angle(trans)
    II4 = double(II4)
    plt.imshow(II4, cmap=plt.get_cmap('gray'))
    #plt.show()

    uf = uf * trans
    u1 = ifft2(uf, (N1, M1))
    u1 = ifftshift(u1)
    u1 = u1 * C
    u1_crop = u1[int(N1/2 - N4/2):int(N1/2 + N4/2), int(M1/2 - M4/2):int(M1/2 + M4/2)]

    # add tukey_window filter
    tukey_win = sqrt(outer(signal.tukey(N4, alpha=0.08), signal.tukey(M4, alpha=0.08)))
    II4 = angle(trans)
    II4 = double(II4)
    plt.imshow(II4, cmap=plt.get_cmap('gray'))
    #plt.show()
    #u1_crop = u1_crop * tukey_win
    return u1_crop


def inverse_ARSS_frenel_function(u, z):
    m = arange(-M1 / 2, M1 / 2)
    n = arange(-N1 / 2, N1 / 2)

    mm, nn = meshgrid(m, n)

    s = 1/3  # 缩放参数
    t = 1/3

    ox = 8e-6 * 0  # 平移参数
    oy = 8e-6 * 0

    dx0 = 8e-6 * 3
    dy0 = 8e-6 * 3

    xm0 = dx0 * m
    ym0 = dy0 * n
    xx0, yy0 = meshgrid(xm0, ym0)

    dx1 = 8e-6 * 3
    dy1 = 8e-6 * 3

    xm1 = dx1 * m
    ym1 = dy1 * n
    xx1, yy1 = meshgrid(xm1, ym1)

    u_patch = zeros((N1, M1), dtype=complex)  # 扩大物平面
    u_patch[int(N1/2 - N4/2):int(N1/2 + N4/2), int(M1/2 - M4/2):int(M1/2 + M4/2)] = u

    # 物光乘以收敛光
    u_c = u_patch  # * c_light
    rect = logical_and(abs(mm) < M/2, abs(nn) < N/2)

    # 常数相位
    C = exp(j1 * h * z + j1 * pi / (h * z) * (
                ((1 - s) * xx1 ** 2 + 2 * ox * xx1 + ox ** 2) + ((1 - t) * yy1 ** 2 + 2 * oy * yy1 + oy ** 2))) / (
                    j1 * h * z) #* rect
    u_re = u_patch * exp(j1 * pi / (h*z) * (((s**2 - s)*xx0**2 - 2 * s * ox * xx0) + ((t**2 - t)*yy0**2 - 2 * t * oy * yy0))) #* rect
    uf = fft2(u_re, (N1, M1))
    uf = fftshift(uf)

    trans = exp(j1 * pi * ((s * xx1 ** 2) + (t * yy1 ** 2))/(h*z))
    rect2 = logical_and(abs(mm) < M, abs(nn) < N)
    trans = trans #* rect2

    trans = fft2(trans, (N1, M1))
    trans_abs = amax(abs(trans))
    #trans = trans_abs * exp(j1 * angle(trans))

    uf = uf * trans
    u1 = ifft2(uf, (N1, M1))
    u1 = ifftshift(u1)
    u1 = u1 * C
    u1_crop = u1[int(N1/2 - N4/2):int(N1/2 + N4/2), int(M1/2 - M4/2):int(M1/2 + M4/2)]
    return u1_crop


def inverse_angular_spectrum_function(z, u, k):
    """
    角谱衍射变换 DFFT
    :param z: 衍射距离
    :param u: 输入物函数
    :param M: 图像宽度
    :param N: 图像高度
    :param k: 光波长
    :return: 角谱衍射结果
    """
    u_patch = zeros((N2, M2), dtype=complex)  # 扩大物平面
    u_patch[int(N2/2 - N4/2):int(N2/2 + N4/2), int(M2/2 - M4/2):int(M2/2 + M4/2)] = u

    m = arange(-M2 / 2, M2 / 2)
    n = arange(-N2 / 2, N2 / 2)
    fx = 1 / (dx * M2)  # 空间頻域坐标
    fy = 1 / (dy * N2)

    # band limit
    fx_limit = 1 / (sqrt((2 * fx * z) ** 2 + 1) * h)
    fy_limit = 1 / (sqrt((2 * fy * z) ** 2 + 1) * h)

    fx = fx * m
    fy = fy * n
    xx, yy = meshgrid(fx, fy)
    Uf = fft2(u_patch, (N2, M2))
    # Uf = fftshift(Uf)  # 数字模拟加了fftshift就会得到错误的结果
    trans = exp(j1 * k * z * sqrt(1 - (h * xx) ** 2 - (h * yy) ** 2))

    rect = logical_and(abs(xx) < fx_limit/1, abs(yy) < fy_limit/1)
    #trans = trans * rect

    f2 = Uf * trans
    Uf = ifft2(f2, (N2, M2))
    return Uf


def add_blazeG(t=2, b=1, c=0):
    """
    叠加的闪耀光栅相位，需要之前已经进行了取相位操作
    b,c代表闪耀光栅的方向. b=0,c=1代表垂直闪耀光栅，反之亦然. b,c还可以取小数
    :param t: 闪耀光栅的周期
    :param b: 闪耀光栅的方向
    :param c: 闪耀光栅的方向
    :return:
    """
    m = arange(-M / 2, M / 2)
    n = arange(-N / 2, N / 2)
    mm, nn = meshgrid(m, n)
    d_blazedg_phase = 2 * pi / t * mod(b * mm + c * nn, t)
    return d_blazedg_phase


if __name__ == '__main__':
    M = 1920
    N = 1080
    S_max = max(M, N)
    S_min = min(M, N)
    M1 = int(2*M)
    N1 = int(2*N)

    S = 4  # zero padding origin image, scale factor
    if M*S % 2 == 0:
        M2 = int(M*S)
    else:
        M2 = int(M*S) - 1
    if N*S % 2 == 0:
        N2 = int(N*S)
    else:
        N2 = int(N*S) - 1

    S2 = 1
    M4 = int(S2*M)
    N4 = int(S2*N)

    h = 0.532e-6  # m
    j1 = (-1) ** 0.5
    k = 2 * pi / h
    z1 = 0.9
    z2 = 0.21
    T = 2  # 不同的T对应不同的闪耀角

    # SLM pix 像素
    dx = 6.4e-6  # m
    dy = 6.4e-6  # m

    m = arange(-M / 2, M / 2)
    n = arange(-N / 2, N / 2)

    m_xx, m_yy = meshgrid(m, n)
    c_xx, c_yy = meshgrid(m*dx, n*dy)

    # 正透镜相位因子
    s_phase = pi *5/ (h * z1) * (c_xx ** 2 + c_yy ** 2)

    # 读取图片
    default_dir = r"C:\Users"
    fname = tk.filedialog.askopenfilename(title=u"选择文件", initialdir=(os.path.expanduser(default_dir)))
    with Image.open(fname) as img:
        img = img.resize((M, N))
    # 分离三个颜色通道
    red, green, blue, alpha = img.split()
    gray = img.convert('L')
    # 将三个通道保存为numpy矩阵
    gray = array(gray)
    green = array(green)
    blue = array(blue)
    red = array(red)
    # 设置要计算的通道
    u0 = gray

    # 输入物函数相位
    uniform_phase = ones((N, M))*pi*0
    u0_replace = u0 * exp(j1*uniform_phase)

    # 需要叠加的竖直闪耀光栅相位
    v_d_blazedg_phase = add_blazeG(b=0, c=1)

    # checkerboard
    M3 = int(M/2)
    N3 = int(N/2)
    Mask1 = array(([1, 0] * M3 + [0, 1] * M3) * N3).reshape((N, M))
    Mask2 = 1 - Mask1

    # 程序有效开始时间
    start_time = time.time()

    # ARSS fresnel diffraction
    u_ARSS = ARSS_frenel_function(u0_replace, z1+z2)

    # 程序有效结束时间
    end_time = time.time()

    # double-phase method
    uf_max_ini = amax(abs(u_ARSS))
    uf_amp = abs(u_ARSS)/uf_max_ini*2
    uf_ang = angle(u_ARSS)
    uf_max = amax(abs(uf_amp))
    theta1 = uf_ang + arccos(uf_amp/uf_max)
    theta2 = uf_ang - arccos(uf_amp/uf_max)
    theta1_norm = theta1 + pi/2*3  # 两个必须加上一样的相位偏移
    theta2_norm = theta2 + pi/2*3
    theta = (theta1_norm * Mask1 + theta2_norm * Mask2) + v_d_blazedg_phase
    theta = mod(theta, 2*pi) * 255 / (2 * pi)
    # theta = 2*pi - theta

    u_d = exp(j1*theta)

    # 矩形滤波器
    x_m_r = arange(-S_max / 2, S_max / 2)
    y_m_r = arange(-S_max / 2, S_max / 2)
    mr_xx, mr_yy = meshgrid(x_m_r, y_m_r)
    c_mask_filter2 = logical_and(abs(mr_xx) < 600, abs(mr_yy) < 350)

    II2 = abs(c_mask_filter2)
    II2 = double(II2)
    plt.imshow(II2, cmap=plt.get_cmap('gray'))
    plt.show()

    # 傅里叶变换滤波
    u_d_padding = zeros((S_max, S_max), dtype=complex)
    u_d_padding[int(S_max / 2 - N / 2):int(S_max / 2 + N / 2), int(S_max / 2 - M / 2):int(S_max / 2 + M / 2)] = u_d
    u_d_f = fftshift(fft2(u_d_padding)) * c_mask_filter2
    u0_recover = ifft2(u_d_f)
    u0_recover = u0_recover[int(S_max / 2 - N / 2):int(S_max / 2 + N / 2), int(S_max / 2 - M / 2):int(S_max / 2 + M / 2)]

    # 重建图像
    u0_recover1 = inverse_ARSS_frenel_function(u0_recover, -(z1+z2))
    # u0_recover2 = inverse_angular_spectrum_function((z1+z2), u0_recover, -k)
    II3 = abs(u_d_f)
    II3 = double(II3)
    plt.imshow(II3, cmap=plt.get_cmap('gray'))
    plt.show()

    II4 = abs(u0_recover1)
    II4 = double(II4)
    plt.imshow(II4, cmap=plt.get_cmap('gray'))
    plt.show()

    # 保存全息图
    im1 = Image.fromarray(uint8(theta))
    # im1.save('/root/PycharmProjects/Experiment/Experiment Picture/Generated/clock_532_s-3_z1_0-9_z2_0-2-1_b-0_c-1.bmp')
