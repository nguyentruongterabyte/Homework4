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
# print(I1)
Itilde1 = fftshift(fft2(I1))
print("Re[DFT(I1)]: ")
print(np.real(Itilde1).round(4))

print("Im[DFT(I1)]:")
print(np.imag(Itilde1).round(4))

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(np.real(I1), cmap='gray', vmin=-1, vmax=1)
plt.title("Real Part of I1")
plt.axis('off')
plt.subplot(122)
plt.imshow(np.imag(I1), cmap='gray', vmin=-1, vmax=1)
plt.title("Imaginary Part of I1")
plt.axis('off')
plt.show()
