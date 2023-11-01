import numpy as np
import matplotlib.pyplot as plt

image_size = (8, 8)

u1 = 1.5
v1 = 1.5

COLS, ROWS = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

# print(COLS)
# print(ROWS)

I5 = np.cos(2 * np.pi * (u1 * COLS + v1 * ROWS) / 8)

print("Real Part of I5: ")
print(np.real(I5).round(4))

print("Imaginary Part of I5: ")
print(np.imag(I5).round(4))

plt.figure(figsize=(6, 6))
plt.imshow(np.real(I5), cmap='gray', vmin=-1, vmax=1)
plt.title("I5 as an 8 bpp Gray Scale Image")
plt.axis('off')
plt.show()
