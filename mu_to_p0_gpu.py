import numpy as np
from numba import jit, cuda, float32
import math


@cuda.jit
def mu_to_p0_kernel(mu, mu_background, source, h, xp, yp, result, a, dpx, dpy, fluence):
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
            else:
                cuda.atomic.add(a, (ty, tx), mu_background * h)
                

    cuda.syncthreads()

    if tx < mu.shape[1] and ty < mu.shape[0]:
        result[ty, tx] = mu[ty, tx] * math.exp(-a[ty, tx]) * ((math.pi * d**2) ** -1)
        fluence[ty, tx] = math.exp(-a[ty, tx]) * ((math.pi * d**2) ** -1)

def mu_to_p0_gpu(mu, mu_background, source, h, xp, yp):
    assert mu.shape[1] == xp.shape[0]
    assert mu.shape[0] == yp.shape[0]

    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]

    a = np.zeros_like(mu)
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    
    dev_mu = cuda.to_device(mu)
    dev_a = cuda.to_device(a)
    dev_p0 = cuda.to_device(p0)
    dev_fluence = cuda.to_device(fluence)
    dev_xp = cuda.to_device(xp)
    dev_yp = cuda.to_device(yp)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (mu.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (mu.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    mu_to_p0_kernel[blocks_per_grid, threads_per_block](dev_mu, mu_background, source, h, dev_xp, dev_yp, dev_p0, dev_a, dpx, dpy, dev_fluence)

    return dev_p0.copy_to_host(), dev_a.copy_to_host(), dev_fluence.copy_to_host()



#####################################################



@cuda.jit
def mu_to_p0_3d_kernel(mu, mu_background, source, h, xp, yp, zp, result, a, dpx, dpy, dpz, fluence):
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
            else:
                cuda.atomic.add(a, (tz, ty, tx), mu_background * h)
            

    cuda.syncthreads()

    if tx < mu.shape[2] and ty < mu.shape[1] and tz < mu.shape[0]:
        result[tz, ty, tx] = mu[tz, ty, tx] * math.exp(-a[tz, ty, tx]) * ((math.pi * d**3) ** -1)
        fluence[tz, ty, tx] =  math.exp(-a[tz, ty, tx]) * ((math.pi * d**3) ** -1)

def mu_to_p0_3d_gpu(mu, mu_background, source, h, xp, yp, zp):
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]

    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    dpz = zp[1] - zp[0]
    
    dev_mu = cuda.to_device(mu)
    dev_source = cuda.to_device(source)
    dev_xp = cuda.to_device(xp)
    dev_yp = cuda.to_device(yp)
    dev_zp = cuda.to_device(zp)
    dev_fluence = cuda.to_device(np.zeros_like(mu, dtype = np.float32))
    dev_result = cuda.to_device(np.zeros_like(mu, dtype=np.float32))
    dev_a = cuda.to_device(np.zeros_like(mu, dtype=np.float32))

    threads_per_block = (8, 8, 8)  # Adjust based on GPU ability
    blocks_per_grid_x = (mu.shape[2] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (mu.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (mu.shape[0] + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    mu_to_p0_3d_kernel[blocks_per_grid, threads_per_block](
        dev_mu, mu_background, dev_source, h, dev_xp, dev_yp, dev_zp, dev_result, dev_a, dpx, dpy, dpz, dev_fluence
    )
    
    P0 = dev_result.copy_to_host()
    a = dev_a.copy_to_host()

    return P0, a, dev_fluence.copy_to_host()

#####################################################


@cuda.jit
def mu_to_p0_cone_kernel(mu, mu_background, source, h, xp, yp, theta, direction, result_p0, result_a, mask, dpx, dpy, fluence):
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
                else:
                    cuda.atomic.add(result_a, (ty, tx), mu_background * h)

    cuda.syncthreads()

    if tx < mu.shape[1] and ty < mu.shape[0]:
        result_p0[ty, tx] = mask[ty, tx] * mu[ty, tx] * math.exp(-result_a[ty, tx]) * ((np.pi * d**2)**-1)
        fluence[ty, tx] = mask[ty, tx] *  math.exp(-result_a[ty, tx]) * ((np.pi * d**2)**-1)
    

def mu_to_p0_cone_gpu(mu, mu_background, source, h, xp, yp, theta, direction):
    assert mu.shape[1] == xp.shape[0]
    assert mu.shape[0] == yp.shape[0]
    
    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    
    dev_mu = cuda.to_device(mu)
    dev_source = cuda.to_device(source)
    dev_xp = cuda.to_device(xp)
    dev_yp = cuda.to_device(yp)
    dev_result_p0 = cuda.to_device(np.zeros_like(mu, dtype=np.float32))
    dev_result_a = cuda.to_device(np.zeros_like(mu, dtype=np.float32))
    dev_fluence = cuda.to_device(np.zeros_like(mu, dtype=np.float32))
    dev_mask = cuda.to_device(np.zeros_like(mu, dtype=np.int32))

    threads_per_block = (16, 16)  # Adjust block size based on your GPU's capabilities
    blocks_per_grid_x = (mu.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (mu.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    mu_to_p0_cone_kernel[blocks_per_grid, threads_per_block](
        dev_mu, mu_background, dev_source, h, dev_xp, dev_yp, theta, direction, dev_result_p0, dev_result_a, dev_mask, dpx, dpy, dev_fluence
    )

    return dev_result_p0.copy_to_host(), dev_result_a.copy_to_host(), dev_fluence.copy_to_host()


#####################################################


@cuda.jit
def mu_to_p0_cone_3d_kernel(mu, mu_background, source, h, xp, yp, zp, direction_vector, theta, a, p0, dpx, dpy, dpz, fluence):
    xs, ys, zs = source
    index_x, index_y, index_z = cuda.grid(3)
    
    if (index_x < mu.shape[2]) and (index_y < mu.shape[1]) and (index_z < mu.shape[0]):
        xi = xp[index_x]
        yi = yp[index_y]
        zi = zp[index_z]

        #vector shift
        dx = xi - xs
        dy = yi - ys
        dz = zi - zs

        dot_product = direction_vector[0] * dx + direction_vector[1] * dy + direction_vector[2] * dz
        
        dir_magnitude = (direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2)**0.5
        shift_magnitude = (dx**2 + dy**2 + dz**2)**0.5

        if dot_product > 0.0 and dir_magnitude > 0.0 and shift_magnitude > 0.0:
            point_angle = np.arccos(dot_product / (dir_magnitude * shift_magnitude))
        else:
            point_angle = 0.0

        if point_angle <= theta / 2:
            d = shift_magnitude  # Use the precomputed magnitude
            n = int(d / h) + 1
            dx /= (n - 1)
            dy /= (n - 1)
            dz /= (n - 1)

            for point_i in range(n):
                i_x = int((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx)
                i_y = int((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy)
                i_z = int((zs + point_i * dz - zp[0] + 0.51 * dpz) / dpz)

                if (0 <= i_x < mu.shape[2]) and (0 <= i_y < mu.shape[1]) and (0 <= i_z < mu.shape[0]):
                    a[index_z, index_y, index_x] += mu[i_z, i_y, i_x] * h
                else: 
                    a[index_z, index_y, index_x] += mu_background * h
                    
            p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * math.exp(-a[index_z, index_y, index_x]) * ((np.pi * d**3)**-1)
            fluence[index_z, index_y, index_x] = math.exp(-a[index_z, index_y, index_x]) * ((np.pi * d**3)**-1)

def mu_to_p0_cone_3d_gpu(mu, mu_background, source, h, xp, yp, zp, direction_vector, theta):

    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]

    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    dpz = zp[1] - zp[0]

    dev_mu = cuda.to_device(mu)
    dev_source = cuda.to_device(source)
    dev_xp = cuda.to_device(xp)
    dev_yp = cuda.to_device(yp)
    dev_zp = cuda.to_device(zp)
    dev_a = cuda.to_device(np.zeros_like(mu, dtype=np.float32))
    dev_p0 = cuda.to_device(np.zeros_like(mu, dtype =np.float32))
    dev_fluence = cuda.to_device(np.zeros_like(mu, dtype =np.float32))
    dev_direction_vector = cuda.to_device(direction_vector)
    
    

    threadsperblock = (16, 16, 1)
    blockspergrid_x = (mu.shape[2] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (mu.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (mu.shape[0] + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    mu_to_p0_cone_3d_kernel[blockspergrid, threadsperblock](dev_mu, mu_background, dev_source, h, dev_xp, dev_yp, dev_zp, dev_direction_vector, theta, dev_a, dev_p0, dpx, dpy, dpz, dev_fluence)
    

    return dev_p0.copy_to_host(), dev_a.copy_to_host(), dev_fluence.copy_to_host()

################################################################

@cuda.jit
def mu_to_p0_wedge_3d_kernel(mu, mu_background, source_start, source_end, ray_direction, theta, h, xp, yp, zp, a, p0, fluence, dpx, dpy, dpz):
    xs, ys, zs = source_start
    xe, ye, ze = source_end
    zs_pixel = int((zs - zp[0] + .51 * dpz) / dpz)
    ze_pixel = int((ze - zp[0] + .51 * dpz) / dpz)
    index_x, index_y, index_z = cuda.grid(3)

    if (index_x < mu.shape[2]) and (index_y < mu.shape[1]) and (index_z < mu.shape[0]) and min(zs_pixel, ze_pixel) < index_z <  max(zs_pixel, ze_pixel):
        xi = xp[index_x]
        yi = yp[index_y]
        zi = zp[index_z]
    
        point_angle = math.atan2(float32(yi - ys), float32(xi - xs)) - ray_direction

        if abs(point_angle) <= theta / 2:
            
            d = math.sqrt(float32(xi - xs)**2 + float32(yi-ys)**2)

            n = int(d / h) + 1
            dx = (xi - xs) / (n - 1)
            dy = (yi - ys) / (n - 1)

            for point_i in range(n):
                i_x = int((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx)
                i_y = int((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy)

                if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1]:
                    a[index_z, index_y, index_x] += mu[index_z, i_y, i_x] * h
                else:
                    a[index_z, index_y, index_x] += mu_background * h

            p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * math.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)
            fluence[index_z, index_y, index_x] = math.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)

def mu_to_p0_wedge_3d_gpu(mu, mu_background, source_start, source_end, ray_direction, theta, h, xp, yp, zp):
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]

    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    dpz = zp[1] - zp[0]

    a = np.zeros_like(mu, dtype=np.float32)
    fluence = np.zeros_like(mu, dtype=np.float32)
    p0 = np.zeros_like(mu, dtype=np.float32)

    dev_mu = cuda.to_device(mu)
    dev_a = cuda.to_device(a)
    dev_fluence = cuda.to_device(fluence)
    dev_p0 = cuda.to_device(p0)
    dev_xp = cuda.to_device(xp)
    dev_yp = cuda.to_device(yp)
    dev_zp = cuda.to_device(zp)
    dev_source_start = cuda.to_device(source_start)
    dev_source_end = cuda.to_device(source_end)

    threadsperblock = (16, 16, 1)
    blockspergrid_x = (mu.shape[2] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (mu.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (mu.shape[0] + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    print(source_start)
    mu_to_p0_wedge_3d_kernel[blockspergrid, threadsperblock](
        dev_mu, mu_background, dev_source_start, dev_source_end, ray_direction, theta, h, dev_xp, dev_yp, dev_zp, dev_a, dev_p0, dev_fluence, dpx, dpy, dpz
    )

    return dev_p0.copy_to_host(), dev_a.copy_to_host(), dev_fluence.copy_to_host()

############################################################################################################

@cuda.jit
def mu_to_p0_wedge_variable_beam_3d_kernel(
    mu, mu_background, xs, ys, zs, xe, ye, ze, ray_direction, theta, h, xp, yp, zp, I, a, p0, fluence
):
    tx, ty, tz = cuda.grid(3)
    
    if tx < mu.shape[2] and ty < mu.shape[1] and tz < mu.shape[0]:
        xi = xp[tx]
        yi = yp[ty]
        zi = zp[tz]
        
        zs_pixel = int((zs - zp[0] + 0.51 * (zp[1] - zp[0])) / (zp[1] - zp[0]))
        ze_pixel = int((ze - zp[0] + 0.51 * (zp[1] - zp[0])) / (zp[1] - zp[0]))
        
        if zs_pixel <= tz < ze_pixel:
            point_angle = math.atan2(yi - ys, xi - xs)
            
            theta_start = ray_direction - (theta / 2)
            theta_end = ray_direction + (theta / 2)
            d_theta = theta / (len(I[tz]) - 1)
            
            if theta_start <= point_angle <= theta_end:
                I_section = I[tz]
                clockwise_ray_i = int(math.ceil((point_angle - theta_start) / d_theta))
                countercw_ray_i = int(math.floor((point_angle - theta_start) / d_theta))
                
                clockwise_ray_theta = theta_start + (clockwise_ray_i * d_theta)
                countercw_ray_theta = theta_start + (countercw_ray_i * d_theta)
                
                clockwise_ray_Intensity = I_section[clockwise_ray_i]
                countercw_ray_Intensity = I_section[countercw_ray_i]
                
                ray_intensity = (
                    (abs(point_angle - countercw_ray_theta) / d_theta) * clockwise_ray_Intensity
                    + (abs(point_angle - clockwise_ray_theta) / d_theta) * countercw_ray_Intensity
                )
                

                d = math.sqrt((xi - xs) ** 2 + (yi - ys) ** 2)
                n = int(d / h) + 1
                dx = (xi - xs) / (n - 1)
                dy = (yi - ys) / (n - 1)
                dz = (zi - zi) / (n - 1)
                
                for point_i in range(n):
                    i_x = int((xs + point_i * dx - xp[0] + 0.51 * (xp[1] - xp[0])) / (xp[1] - xp[0]))
                    i_y = int((ys + point_i * dy - yp[0] + 0.51 * (yp[1] - yp[0])) / (yp[1] - yp[0]))

                    if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1]:
                        cuda.atomic.add(a, (tz, ty, tx), mu[tz, i_y, i_x] * h)
                    else:
                        cuda.atomic.add(a, (tz, ty, tx), mu_background * h)
                
                p0[tz, ty, tx] = mu[tz, ty, tx] * math.exp(-a[tz, ty, tx]) * ((math.pi * (d ** 3)) ** -1)
                fluence[tz, ty, tx] = math.exp(-a[tz, ty, tx]) * ((math.pi * (d ** 3)) ** -1)

def mu_to_p0_wedge_variable_beam_3d_gpu(
    mu, mu_background, source_start, source_end, ray_direction, theta, h, xp, yp, zp, I
):
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]

    xs, ys, zs = source_start
    xe, ye, ze = source_end

    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    dpz = zp[1] - zp[0]

    a = np.zeros_like(mu, dtype=np.float32)
    p0 = np.zeros_like(mu, dtype=np.float32)
    fluence = np.zeros_like(mu, dtype=np.float32)

    threads_per_block = (8, 8, 8)  # Adjust based on GPU ability
    blocks_per_grid_x = (mu.shape[2] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (mu.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (mu.shape[0] + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    dev_mu = cuda.to_device(mu)
    dev_xp = cuda.to_device(xp)
    dev_yp = cuda.to_device(yp)
    dev_zp = cuda.to_device(zp)
    dev_I = cuda.to_device(I)
    dev_a = cuda.to_device(a)
    dev_p0 = cuda.to_device(p0)
    dev_fluence = cuda.to_device(fluence)

    mu_to_p0_wedge_variable_beam_3d_kernel[blocks_per_grid, threads_per_block](
        dev_mu, mu_background, xs, ys, zs, xe, ye, ze, ray_direction, theta, h, dev_xp, dev_yp, dev_zp, dev_I, dev_a, dev_p0, dev_fluence
    )

    return dev_p0.copy_to_host(), dev_a.copy_to_host(), dev_fluence.copy_to_host()