import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0
import inverse_problem


def create_blood_vessel_tube(xp, yp, zp, radius, length): 
    nx, ny, nz = xp.shape[0], yp.shape[0], zp.shape[0]
    dz = zp[1] - zp[0]
    phantom = np.zeros((nx, ny, nz))

    center_x = (max(xp) + min(xp)) / 2
    center_y = (max(yp) + min(yp)) / 2
    
    z_start = (len(zp) // 2 ) - int((np.floor((length / 2) / dz)))
    z_end = (len(zp) // 2 ) + int((np.floor((length / 2) / dz)))

    for i in range(nx):
        for j in range(ny):
            for k in range(z_start, z_end):
                xi = xp[i] #physical coordinates
                yi = yp[j]

                
                distance = np.sqrt((xi - center_x)**2 + (yi - center_y)**2)
                
                if distance < radius:
                    phantom[i, j, k] = 1
                    
    phantom = np.transpose(phantom, (2, 1, 0))
    return phantom

def create_gaussian_array(size, center, sigma): # for creating I
    x = np.arange(size)
    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return gaussian


Lx = 1
Ly = 1
Lz = 3

nx = 20
ny = 20
nz = 60

dx = Lx / nx
dy = Ly / ny
dz = Lz / nz

xc = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
yc = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
zc = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, nz)

length = Lz
radius = 0.1
mu = create_blood_vessel_tube(xc, yc, zc, radius, length)
mu_background = .2

source_start = np.array([-1, 0, -1]) #vertical source, xyz
source_end = np.array([-1, 0, 1])

#ray direction is defined as the rotational angle around the z-axis in radians
direction_vector = [1,0,0]
theta = np.pi/4

I_control = np.ones(11)
I = create_gaussian_array(11, 6, 5)

P0_original, a_original, fluence_original = mu_to_p0.mu_to_p0_cone_stacked_cone (mu, mu_background, source_start, source_end, dx/2, xc, yc, zc, direction_vector, theta, I)
#P0, a, fluence = mu_to_p0.mu_to_p0_cone_stacked_cone (mu, mu_background, source_start, source_end, dx/2, xc, yc, zc, direction_vector, theta, I_control)
shortFat_H = inverse_problem.p0_to_H_stackedCone(mu, mu_background, dx/2, source_start, source_end, xc, yc, zc, direction_vector, theta, I_control)


tall_I = np.zeros((len(I) * nz))
for z in range(nz):
    tall_I[z * len(I): (z * len(I)) + len(I)] = I


y = shortFat_H.dot(tall_I)

P0_new = y.reshape(nz, ny, -1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].set_title("P0_new")
ax[1].set_title("P0_original")

ax[0].imshow(np.max(P0_new, axis = 1), cmap = "gray")
ax[1].imshow(np.max(P0_original, axis = 1), cmap = "gray")

print(np.allclose(P0_original, P0_new, atol= 0.00001))

plt.show()
