import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def phase_generation_fresnel(u_in, feature_size, wavelength, prop_dist):
    field_resolution = u_in.shape
    num_y, num_x = field_resolution[0], field_resolution[1]
    dy, dx = feature_size
    z = prop_dist
    m = np.arange(-num_y / 2, num_y/ 2)
    n= np.arange(-num_x / 2, num_x / 2)
    y = m * dy
    x = n * dx
    X, Y = np.meshgrid(x, y)
    phaseh = np.exp(1j * np.pi / (wavelength * z) * (X ** 2 + Y ** 2))
    du=wavelength*z/num_x/dx
    dv=wavelength*z/num_y/dy
    u=m*abs(du)
    v=n*abs(dv)
    U,V=np.meshgrid(u,v)
    phasev=np.exp(1j*np.pi/wavelength/z*(U**2+V**2))
    return phaseh, phasev,dv

def propration(u_in, phaseh, phasev,direction='forward'):
    if direction=='forward':
        U = u_in * phaseh
        U = np.fft.fft2(U)
        U = np.fft.fftshift(U)
        U = U * phasev  
    else:
        U = u_in * phasev
        U = np.fft.ifft2(U)
        U = np.fft.ifftshift(U)
        U = U * phaseh

    return U

if __name__ == '__main__':
    path='/home/nil/repoartical/speckleReduced/jpg/image2.jpg'
    img = Image.open(path).convert('L').resize((1080, 1080))
    img = np.array(img)
    img = img / 255
    
    mm, um, nm = 1e-3, 1e-6, 1e-9
    wavelength = 532 * nm
    feature_size = (8 * um, 8 * um)
    prop_dist = 0.5
    phaseh, phasev,dv = phase_generation_fresnel(img, feature_size, wavelength, prop_dist)
    rand_phase = np.random.rand(1080, 1080)
    u_in= img * np.exp(1j * np.pi * rand_phase)
    U = propration(u_in, phaseh, phasev,'forward')

    plt.figure()
    plt.imshow(np.abs(U), cmap='gray')
    plt.show()
    U_phase = np.angle(U)
    U_phase=np.exp(1j*U_phase*2*np.pi)
    dv_size=(dv, dv)
    phaseh, phasev,dv = phase_generation_fresnel(img, feature_size, wavelength, -prop_dist)
    U = propration(U_phase, phaseh, phasev,'backward')
    plt.figure()
    plt.imshow(np.abs(U), cmap='gray')
    plt.show()

