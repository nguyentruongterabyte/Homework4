"""
  Họ và tên: Nguyễn Thái Trưởng
  MSSV     : N20DCCN083
"""

import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

image_size = (8, 8)

u0 = 2
v0 = 2

COLS, ROWS = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

# print(COLS)
# print(ROWS)

I1 = 0.5 * np.exp(1j * 2 * np.pi / 8 * (u0 * COLS + u0 * ROWS))
I2 = 0.5 * np.exp(-1j * 2 * np.pi / 8 * (u0 * COLS + u0 * ROWS))

I4 = -1j * (I1 - I2)

Itilde4 = fftshift(fft2(I4))

print("Re[DFT(I4)]:")
print(np.real(Itilde4).round(4))

print("Im[DFT(I4)]:")
print(np.imag(Itilde4).round(4))

plt.figure(figsize=(6, 6))
plt.imshow(np.imag(I1), cmap='gray', vmin=-1, vmax=1)
plt.title("I4 as an 8 bpp Gray Scale Image (Imaginary Part of I1)")
plt.axis('off')
plt.show()
