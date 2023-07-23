import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0


def gauss_density_pattern(size_x, size_y, amplitude, sigma):
    x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))
    x0, y0 = size_x // 2, size_y // 2  # Center of the ball
    density = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return np.array(density)

mu = gauss_density_pattern(100, 100, 50, 20)

            
P0, P = (mu_to_p0.mu_to_p0(mu, (101,101), 1)) # source point at bottom right of image


fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].set_title("Attenuation Matrix")
ax[1].set_title("'P'")
ax[2].set_title("P0")
ax[0].imshow(mu)
ax[1].imshow(P)
ax[2].imshow(P0)
plt.show()