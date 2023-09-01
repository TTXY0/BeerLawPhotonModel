import numpy as np
from numba import jit, cuda

import numpy as np
import math
from numba import cuda

@cuda.jit
def mu_to_p0_kernel(mu, source, h, xp, yp, result, a, dpx, dpy):
    tx, ty = cuda.grid(2)
    xs, ys = source

    if tx < mu.shape[1] and ty < mu.shape[0]:
        xi = xp[tx]
        yi = yp[ty]

        d = ((xi - xs) ** 2 + (yi - ys) ** 2) ** 0.5
        n = int(d / h) + 1

        dx = (xi - xs) / (n - 1)
        dy = (yi - ys) / (n - 1)

        for point_i in range(n):
            i_x = int(math.floor((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
            i_y = int(math.floor((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy))

            if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]:
                cuda.atomic.add(a, (ty, tx), mu[i_y, i_x] * h) 

    cuda.syncthreads()

    if tx < mu.shape[1] and ty < mu.shape[0]:
        result[ty, tx] = mu[ty, tx] * math.exp(-a[ty, tx])

def mu_to_p0_gpu(mu, source, h, xp, yp):
    xs, ys = source
    assert mu.shape[1] == xp.shape[0]
    assert mu.shape[0] == yp.shape[0]

    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]

    a = np.zeros_like(mu)
    result = np.zeros_like(mu)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (mu.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (mu.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    mu_to_p0_kernel[blocks_per_grid, threads_per_block](mu, source, h, xp, yp, result, a, dpx, dpy)

    return result, a

@cuda.jit
def mu_to_p0_3d_kernel(mu, source, h, xp, yp, zp, result, a, dpx, dpy, dpz):
    tx, ty, tz = cuda.grid(3)
    xs, ys, zs = source

    if tx < mu.shape[2] and ty < mu.shape[1] and tz < mu.shape[0]:
        xi = xp[tx]
        yi = yp[ty]
        zi = zp[tz]

        d = math.sqrt((xi - xs)**2 + (yi - ys)**2 + (zi - zs)**2)
        n = int(d / h) + 1

        dx = (xi - xs) / (n - 1)
        dy = (yi - ys) / (n - 1)
        dz = (zi - zs) / (n - 1)

        for point_i in range(n):
            i_x = int(math.floor((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
            i_y = int(math.floor((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy))
            i_z = int(math.floor((zs + point_i * dz - zp[0] + 0.51 * dpz) / dpz))

            if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1] and 0 <= i_z < mu.shape[0]:
                cuda.atomic.add(a, (tz, ty, tx), mu[i_z, i_y, i_x] * h)

    cuda.syncthreads()

    if tx < mu.shape[2] and ty < mu.shape[1] and tz < mu.shape[0]:
        result[tz, ty, tx] = mu[tz, ty, tx] * math.exp(-a[tz, ty, tx])

def mu_to_p0_3d_gpu(mu, source, h, xp, yp, zp):
    xs, ys, zs = source
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]

    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    dpz = zp[1] - zp[0]

    a = np.zeros_like(mu)
    result = np.zeros_like(mu)

    threads_per_block = (8, 8, 8)  # Adjust based on GPU ability
    blocks_per_grid_x = (mu.shape[2] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (mu.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (mu.shape[0] + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    mu_to_p0_3d_kernel[blocks_per_grid, threads_per_block](mu, source, h, xp, yp, zp, result, a, dpx, dpy, dpz)

    return result, a

@cuda.jit
def mu_to_p0_cone_kernel(mu, source, h, xp, yp, theta, direction, result_p0, result_a, mask, dpx, dpy):
    tx, ty = cuda.grid(2)
    xs, ys = source
    
    if tx < mu.shape[1] and ty < mu.shape[0]:
        xi = xp[tx]
        yi = yp[ty]
        point_angle = math.atan2(yi - ys, xi - xs) - direction
        
        if abs(point_angle) <= theta / 2:
            mask[ty, tx] = 1
            
            d = math.sqrt((xi - xs)**2 + (yi - ys)**2)
            n = int(d / h) + 1

            dx = (xi - xs) / (n - 1)
            dy = (yi - ys) / (n - 1)

            for point_i in range(n):
                i_x = int(math.floor((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
                i_y = int(math.floor((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy))

                if 0 <= i_x < mu.shape[1] and 0 <= i_y < mu.shape[0]:
                    cuda.atomic.add(result_a, (ty, tx), mu[i_y, i_x] * h)

    cuda.syncthreads()

    if tx < mu.shape[1] and ty < mu.shape[0]:
        result_p0[ty, tx] = mask[ty, tx] * mu[ty, tx] * math.exp(-result_a[ty, tx])

def mu_to_p0_cone_gpu(mu, source, h, xp, yp, theta, direction):
    xs, ys = source
    assert mu.shape[1] == xp.shape[0]
    assert mu.shape[0] == yp.shape[0]
    
    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    
    a = np.zeros_like(mu)
    result_p0 = np.zeros_like(mu)
    result_a = np.zeros_like(mu)
    mask = np.zeros_like(mu)

    threads_per_block = (16, 16)  # Adjust block size based on your GPU's capabilities
    blocks_per_grid_x = (mu.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (mu.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    mu_to_p0_cone_kernel[blocks_per_grid, threads_per_block](
        mu, source, h, xp, yp, theta, direction, result_p0, result_a, mask, dpx, dpy
    )

    return result_p0, result_a, mask