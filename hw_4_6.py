import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def display_image_components(image_name):
    image = np.fromfile(image_name, dtype=np.uint8).reshape(256, 256)

    centered_dft = fftshift(fft2(image))

    log_magnitude = np.log(np.abs(centered_dft) + 1)

    phase = np.angle(centered_dft)

    plt.figure(figsize=(16, 16))

    plt.subplot(321)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image")

    plt.subplot(322)
    plt.imshow(np.real(centered_dft), cmap='gray', vmin=-1000, vmax=1000)
    plt.title("Real Part of Centered DFT")

    plt.subplot(323)
    plt.imshow(np.imag(centered_dft), cmap='gray', vmin=-1000, vmax=1000)
    plt.title("Imaginary Part of Centered DFT")

    plt.subplot(324)
    plt.imshow(log_magnitude, cmap='gray', norm=LogNorm(vmin=1))
    plt.title("Centered DFT Log-Magnitude Spectrum")

    plt.subplot(325)
    plt.imshow(phase, cmap='gray', vmin=-np.pi, vmax=np.pi)
    plt.title("Phase of Centered DFT")

    plt.tight_layout()
    plt.show()

image_filenames = ["camera.bin", "salesman.bin", "head.bin", "eyeR.bin"]

for image_name in image_filenames:
    display_image_components(image_name)
