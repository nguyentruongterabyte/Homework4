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

I2 = 0.5 * np.exp(-1j * 2 * np.pi / 8 * (u0 * COLS + u0 * ROWS))
# print(I1)
Itilde2 = fftshift(fft2(I2))
print("Re[DFT(I2)]: ")
print(np.real(Itilde2).round(4))

print("Im[DFT(I2)]:")
print(np.imag(Itilde2).round(4))

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(np.real(I2), cmap='gray', vmin=-1, vmax=1)
plt.title("Real Part of I2")
plt.axis('off')
plt.subplot(122)
plt.imshow(np.imag(I2), cmap='gray', vmin=-1, vmax=1)
plt.title("Imaginary Part of i2")
plt.axis('off')
plt.show()
