import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.fft import fft2

I6 = np.fromfile("camera.bin", dtype=np.uint8).reshape(256, 256)

DFT_I6 = fft2(I6)
magnitude = np.abs(DFT_I6)
phase = np.angle(DFT_I6)

J1_magnitude = magnitude
J1_phase = np.zeros_like(phase)

J2_magnitude = np.ones_like(magnitude)
J2_phase = phase

plt.figure(figsize=(8, 8))
plt.imshow(J2_phase, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.title("J2 (Phase of I6) as an 8 bpp Grayscale Image")
plt.axis('off')
plt.show()

JJ1 = np.log(J1_magnitude + 1)
plt.figure(figsize=(8, 8))
plt.imshow(JJ1, cmap='gray', norm=LogNorm(vmin=JJ1.min(), vmax=JJ1.max()))
plt.title("JJ1 (Log of J1 Magnitude) as an 8 bpp Grayscale Image")
plt.axis('off')
plt.show()
