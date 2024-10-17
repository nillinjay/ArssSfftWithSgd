import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
    c_light = np.exp(-1j * np.pi * (s ** 2 / (wavelength * (z2 * c_x)) * xx0 ** 2 + s ** 2 / (wavelength * (z2 * c_y)) * yy0 ** 2))
    return c_light


def phase_generation_sfft(u_in, slm_size, wavelength, prop_dist, dtype=np.complex64):
    field_resolution = u_in.shape
    num_y, num_x = field_resolution[1], field_resolution[0]
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

    radius=3.3e-3/2
    rect=np.sqrt(U**2+V**2)<=radius

    return phasev, phaseh2,rect

def propagation_ARSS_sfft(u_in, phaseh, phaseu, phasec, phasev, phaseh2, rect,direction='forward',dtype=np.complex64):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    通过使用FFT传播输入场u_in通过传递函数TF。
    """
   
    if direction == 'forward':

        u = u_in * phaseu
        # u代表了输入场

        U1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(u), axes=(-2, -1), norm='ortho'))

        Trans = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(phaseh), axes=(-2, -1), norm='ortho'))

        U2 = Trans * U1

        u1 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(U2), axes=(-2, -1), norm='ortho'))
        # 下面是我需要修改的
        u_out = u1 
        plt.title('virtual')
        plt.imshow(np.abs(u_out).squeeze(), cmap='gray')
        plt.show()

        u2 = u_out * phasev*rect*phasec

        u_out = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(u2), axes=(-2, -1), norm='ortho'))
        u_out = u_out * phaseh2

    else:
        u=u_in*phaseh2
        u = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(u), axes=(-2, -1), norm='ortho'))
        u = u * phasev*rect
        plt.title('virtual2')
        plt.imshow(np.abs(u).squeeze(), cmap='gray')
        plt.show()

        u=u*phaseu
        U1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(u), axes=(-2, -1), norm='ortho'))
        Trans=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(phaseh), axes=(-2, -1), norm='ortho'))
        U2=U1*Trans
        u1 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(U2), axes=(-2, -1), norm='ortho'))
        u_out = u1 
        u_out=u_out*phasec

                                         

    return u_out

def phase_generation_Arss(u_in, feature_size, wavelength, prop_dist, dtype=np.complex64, arss_s=1):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    feature_size: (dy, dx) size of the feature in the object plane, which is caculated by dv and arss_s
    prop_dist: z1*totals
    """
    field_resolution = u_in.shape
    num_y, num_x = field_resolution[0], field_resolution[1]
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
    phasec = np.exp(1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / 1j 
    phasec = np.array(phasec, dtype=dtype)

    return phaseh, phaseu, phasec
def double_phase(Uf):
    # Uf shape: [batch_size, channels, height (N), width (M)]
    N, M = Uf.shape[-2], Uf.shape[-1]  # Extract height and width

    # Generate coordinate grids
    x = np.arange(M).reshape(1, M).repeat(N, axis=0)
    y = np.arange(N).reshape(N, 1).repeat(M, axis=1)

    # Create Mask1 using cosine squared
    Mask1 = np.cos(np.pi * (x + y) / 2)**2
    Mask2 = 1 - Mask1  # Inverse of Mask1

    # Remove batch and channel dimensions for computation
    #Uf = Uf.squeeze(0).squeeze(0)  # Now Uf has shape [N, M]

    # Compute amplitude and phase
    Uf_P = np.angle(Uf)
    Uf_A = np.abs(Uf)
    w = Uf_A / np.max(Uf_A)

    # Compute theta1 and theta2
    theta1 = Uf_P + np.arccos(w)
    theta2 = Uf_P - np.arccos(w)

    # Combine phases using the masks
    theta = theta1 * Mask1 + theta2 * Mask2

    # Add batch and channel dimensions back
   # theta = theta[np.newaxis, np.newaxis, :, :]  # Shape: [1, 1, N, M]

    return theta

if __name__ == "__main__":
    # Model Parameter
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    image_res = (1080, 1080)
    z1=-0.14
    arss_s=2.32
    wavelength = 532 * nm
    slm_size=   (8 * um, 8 * um)
    slm_pitch = 8 * um
    virtual_pitch=abs(wavelength*z1/slm_pitch/1080)
    virtual_size=(virtual_pitch, virtual_pitch)
    #totals=abs(totals)
    totals=2.5
    z2=-0.5
    feature_size = (totals * 8 * um, totals * 8 * um)

    image=Image.open('/home/nil/CodeAndArticles/ArssSfftWithSgd/jpg/image3.jpg').convert('L').resize((1080,1080))
    image = np.array(image)/255
    plt.title('Target')


    c_light = clight_generation(u_in=image, wavelength=wavelength,totals=totals,z2=z2)
    target_camp = image * c_light 
    phaseh, phaseu, phasec = phase_generation_Arss(u_in=image, feature_size=virtual_size, wavelength=wavelength, prop_dist=z2,arss_s=1/arss_s)
    phasev,phaseh2 ,rect= phase_generation_sfft(u_in=image, slm_size=slm_size, wavelength=wavelength, prop_dist=z1)
    u_out = propagation_ARSS_sfft(target_camp, phaseh, phaseu, phasec, phasev, phaseh2,rect)
    holo=double_phase(u_out)
    #holo=np.mod(holo,2*np.pi)/2/np.pi
    plt.title('Holo')
    plt.imshow(holo.squeeze(), cmap='gray')
    plt.show()


    plt.title('rect')
    plt.imshow(rect, cmap='gray')
    plt.show()
    phaseh, phaseu, phasec = phase_generation_Arss(u_in=u_out, feature_size=feature_size, wavelength=wavelength, prop_dist=-z2,arss_s=arss_s)
    phasev,phaseh2,rect = phase_generation_sfft(u_in=u_out, slm_size=slm_size, wavelength=wavelength, prop_dist=-z1)
    u_out = propagation_ARSS_sfft(np.exp(1j*holo), phaseh, phaseu, phasec, phasev, phaseh2,rect,direction='backward')
    u_out=np.abs(u_out)
    plt.title('Reconstructed')
    plt.imshow(u_out, cmap='gray')
    plt.show()
    SSIM = ssim(u_out, image, data_range=u_out.max() - image.min())
    PSNR = psnr(u_out, image,data_range=u_out.max() - image.min())
    print(abs(z1))
    print(SSIM)
    print(PSNR)