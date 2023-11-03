import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0

#for phantom
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
                if k == 0 : 
                    print(distance, xi , yi, distance < radius)
                
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

I = create_gaussian_array(11, 5, 5)
# I_row = np.ones(11)
# I = np.array([I_row for _ in range(nz)])

print(mu.shape[0], mu.shape[1], mu.shape[2])
print(zc.shape[0], yc.shape[0], xc.shape[0])


#P0, a, fluence = mu_to_p0.mu_to_p0_wedge_variable_beam_3d(mu, mu_background, source_start, source_end, ray_direction, theta, dx/2, xc, yc, zc, I)
P0, a, fluence = mu_to_p0.mu_to_p0_cone_stacked_cone (mu, mu_background, source_start, source_end, dx/2, xc, yc, zc, direction_vector, theta, I)



fig, ax = plt.subplots(3, 3, figsize=(12,7), sharex= True)
ax[0,0].set_title("Along Z axis", fontsize = 20)
ax[0,1].set_title("Along Y axis", fontsize = 20)
ax[0,2].set_title("Along X axis", fontsize = 20)

ax[0,0].imshow(np.max(P0, axis = 0), cmap = 'gray')
ax[0,1].imshow(np.max(P0, axis = 1), cmap = 'gray')

ax[0,2].imshow(np.max(P0, axis = 2), cmap = 'gray')

ax[1,0].imshow(np.max(a, axis = 0), cmap = 'gray')
ax[1,1].imshow(np.max(a, axis = 1), cmap = 'gray')
ax[1,2].imshow(np.max(a, axis = 2), cmap = 'gray')

ax[2,0].imshow(np.max(fluence, axis = 0), cmap = 'gray')
ax[2,1].imshow(np.max(fluence, axis = 1), cmap = 'gray')
ax[2,2].imshow(np.max(fluence, axis = 2), cmap = 'gray')

ax[0,0].set_ylabel("P0", fontsize = 20)
ax[1,0].set_ylabel("Alpha", fontsize = 20)
ax[2,0].set_ylabel("Fluence", fontsize = 20)

print(np.max(P0, axis = 0).shape)
plt.show()