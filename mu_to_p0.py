import numpy as np

# from numba import jit, cuda

def mu_to_p0_isotropic(mu, mu_background, source, h, xp: np.array, yp: np.array): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
    xs, ys = source
    assert mu.shape[1]==xp.shape[0]
    assert mu.shape[0]==yp.shape[0]
    
    dpx  = xp[1]-xp[0] #spacing between sampling points
    dpy = yp[1]-yp[0]
    a = np.zeros_like(mu)
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    for index_y in range(mu.shape[0]):
        for index_x in range(mu.shape[1]):
            xi = xp[index_x] #physical coordinates
            yi = yp[index_y]
            
            
            d = ((xi-xs)**2 + (yi-ys)**2) **0.5 #euclidean distance between source and target
            n = int(d/h) + 1 # of discrete point

            dx =  (xi - xs) / (n - 1) #physical space dx
            dy = (yi - ys) / (n - 1)
                       
            for point_i in range(n):
 
                i_x = int(np.floor( (xs + point_i * dx - xp[0] + 0.51*dpx ) / dpx ) ) #pixel indices
                i_y = int(np.floor( (ys + point_i * dy - yp[0] + 0.51*dpy ) / dpy ) )
                
                if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]:
                    a[index_y, index_x] += mu[i_y,i_x] * h

                else: #mu_background
                    a[index_y, index_x] += mu_background * h

            p0[index_y, index_x] = mu[index_y, index_x] * (np.pi * d**2)**-1 * np.exp(-a[index_y, index_x]) 
            fluence[index_y, index_x] = np.exp(-a[index_y, index_x]) * ((np.pi * (d**2)) ** -1)
    return p0, a, fluence

def mu_to_p0_wedge(mu, mu_background, source, h, xp: np.array, yp: np.array, theta, direction): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
    xs, ys = source                                                                 #theta is the width of the "flashlight", direction is the "pointed" direction of the flashlight
    assert mu.shape[1]==xp.shape[0]
    assert mu.shape[0]==yp.shape[0]
    
    dpx  = xp[1] - xp[0] #spacing between sampling points
    dpy = yp[1] - yp[0]
    
    a = np.zeros_like(mu)
    
    #Mask for "flashlight"
    mask =  np.zeros_like(mu)
    for index_y in range(mask.shape[0]):
        for index_x in range(mask.shape[1]):
            xi = xp[index_x] #physical coordinates
            yi = yp[index_y]
            point_angle = np.arctan2 (yi - ys, xi - xs) - direction
            if np.abs(point_angle) <= theta/2: 
                mask[index_y, index_x] = 1

    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    for index_y in range(mu.shape[0]):
        for index_x in range(mu.shape[1]):
            if mask[index_y , index_x] == 1:
                
                xi = xp[index_x] #physical coordinates
                yi = yp[index_y]
                
                d = ((xi-xs)**2 + (yi-ys)**2) **0.5 #euclidean distance between source and target
                n = int(d/h) + 1 # of discrete point

                dx =  (xi - xs) / (n - 1) #physical space dx
                dy = (yi - ys) / (n - 1)
                
                for point_i in range(n):
                    i_x = int(np.floor( (xs + point_i * dx - xp[0] + 0.51*dpx ) / dpx ) ) #pixel indices
                    i_y = int(np.floor( (ys + point_i * dy - yp[0] + 0.51*dpy ) / dpy ) ) 
                    
                    if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]:
                        #a[index_x, index_x] = .0
                        a[index_y, index_x] += mu[i_y,i_x] * h
                    
                    else: #mu_background
                        a[index_y, index_x] += mu_background * h
                        
                p0[index_y, index_x] = mu[index_y, index_x] * np.exp(-a[index_y, index_x]) * ((np.pi * (d**2)) ** -1)
                fluence[index_y, index_x] = np.exp(-a[index_y, index_x]) * ((np.pi * (d**2)) ** -1)
            
    return p0, a, fluence

def mu_to_p0_isotropic_3d(mu, mu_background, source, h, xp, yp, zp):
    xs, ys, zs = source
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]
    
    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    dpz = zp[1] - zp[0]
    
    a = np.zeros_like(mu)
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    
    for index_z in range(mu.shape[0]):
        for index_y in range(mu.shape[1]):
            for index_x in range(mu.shape[2]):
                xi = xp[index_x]
                yi = yp[index_y]
                zi = zp[index_z]
                
                d = ((xi - xs)**2 + (yi - ys)**2 + (zi - zs)**2)**0.5
                n = int(d / h) + 1
                
                dx = (xi - xs) / (n - 1)
                dy = (yi - ys) / (n - 1)
                dz = (zi - zs) / (n - 1)
                
                for point_i in range(n):
                    i_x = int(np.floor((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
                    i_y = int(np.floor((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy))
                    i_z = int(np.floor((zs + point_i * dz - zp[0] + 0.51 * dpz) / dpz))
                    
                    if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1] and 0 <= i_z < mu.shape[0]:
                        a[index_z, index_y, index_x] += mu[i_z, i_y, i_x] * h
                        
                    else: #mu_background
                        a[index_z, index_y, index_x] += mu_background * h
                        
                p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)
                fluence[index_z, index_y, index_x] = np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)
                        
    return p0, a, fluence

def mu_to_p0_cone_3d(mu, mu_background, source, h, xp: np.array, yp: np.array, zp: np.array, direction_vector, theta):
    xs, ys, zs = source
    
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]
    
    dpx = xp[1] - xp[0]   
    dpy = yp[1] - yp[0] 
    dpz = zp[1] - zp[0]   
    
    a = np.zeros_like(mu)
    
    mask = np.zeros_like(mu)
    for index_z in range(mask.shape[0]):
        for index_y in range(mask.shape[1]):
            for index_x in range(mask.shape[2]):
                xi = xp[index_x]
                yi = yp[index_y]
                zi = zp[index_z]
                
                vector_shift = np.array([xi - xs, yi - ys, zi - zs])
                 
                point_angle = np.arccos(   np.dot(direction_vector, vector_shift)  /  (np.linalg.norm(direction_vector) * np.linalg.norm(vector_shift)   )   )
                if point_angle <= theta / 2:
                    mask[index_z, index_y, index_x] = 1

    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    for index_z in range(mu.shape[0]):
        for index_y in range(mu.shape[1]):
            for index_x in range(mu.shape[2]):
                if mask[index_z, index_y, index_x] == 1:
                    xi = xp[index_x]
                    yi = yp[index_y]
                    zi = zp[index_z]
                    
                    d = ((xi - xs)**2 + (yi - ys)**2 + (zi - zs)**2)**0.5
                    n = int(d / h) + 1
                    
                    dx = (xi - xs) / (n - 1)
                    dy = (yi - ys) / (n - 1)
                    dz = (zi - zs) / (n - 1)
                    
                    for point_i in range(n):
                        i_x = int(np.floor((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
                        i_y = int(np.floor((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy))
                        i_z = int(np.floor((zs + point_i * dz - zp[0] + 0.51 * dpz) / dpz))
                        
                        if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1] and 0 <= i_z < mu.shape[0]:
                            a[index_z, index_y, index_x] += mu[i_z, i_y, i_x] * h
                        else: 
                            a[index_z, index_y, index_x] += mu_background * h
                            
                    p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)
                    fluence[index_z, index_y, index_x] = np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)
            
    return p0, a, fluence

def mu_to_p0_line(mu, mu_background, source_start, source_end, ray_direction, h, xp, yp):
    assert mu.shape[1] == xp.shape[0]
    assert mu.shape[0] == yp.shape[0]
    
    xs, ys = source_start
    xe, ye = source_end
    
    dpx = xp[1] - xp[0]  # spacing between sampling points
    dpy = yp[1] - yp[0]
    
    a = np.zeros_like(mu)
    
    xs_pixel = int(np.floor((xs - xp[0] + 0.51 * dpx) / dpx))
    ys_pixel = int(np.floor((ys - yp[0] + 0.51 * dpy) / dpy))
    xe_pixel = int(np.floor((xe - xp[0] + 0.51 * dpx) / dpx))
    ye_pixel = int(np.floor((ye - yp[0] + 0.51 * dpy) / dpy))
    
    mask = np.zeros_like(mu)
    mask[ys_pixel, xs_pixel] = 1
    mask[ye_pixel, xe_pixel] = 1
    
    for index_y in range(mask.shape[0]):
        for index_x in range(mask.shape[1]):
            source_vector = np.array([xe_pixel - xs_pixel, ye_pixel - ys_pixel])
            point_vector = np.array([index_x - xs_pixel, index_y - ys_pixel])
            
            perp_slope = -(source_vector[0] / source_vector[1])
            b_start = -perp_slope * xs_pixel + ys_pixel
            b_end = -perp_slope * xe_pixel + ye_pixel
            
            x_start = (index_y - b_start) / perp_slope
            x_end = (index_y - b_end) / perp_slope
            
            projection = (np.dot(point_vector, source_vector) / (np.linalg.norm(source_vector) ** 2)) * source_vector
            projection = projection + np.array([xs_pixel, ys_pixel])
            
            #code for printing source start and end on a
            # if 0 <= int(projection[0]) < mu.shape[0] and 0 <= int(projection[1]) < mu.shape[1]:
            #     a[int(projection[1]), int(projection[0])] = 111
            
            unit_vector = np.array([index_x - projection[0], index_y - projection[1]])
            
            if np.allclose(unit_vector, np.array([0, 0])):
                continue
            else:
                unit_vector = (unit_vector / np.linalg.norm(unit_vector))
            
            if not np.allclose(unit_vector, ray_direction, rtol=0.2) or not np.min([x_end, x_start]) <= projection[0] <= np.max([x_start, x_end]):
                continue
            else:
                mask[index_y, index_x] = 1
                
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    for index_y in range(mu.shape[0]):
        for index_x in range(mu.shape[1]):
            if mask[index_y, index_x] == 1:
                xi = xp[index_x]  # physical coordinates
                yi = yp[index_y]
                
                
                xs_pixel = int((xs - xp[0]) / dpx) #start of source
                ys_pixel = int((ys - yp[0]) / dpy)
                
                xe_pixel = int((xe - xp[0]) / dpx) #end of source
                ye_pixel = int((ye - yp[0]) / dpy)
                
                #Code for printing source start and end on a
                # a[ys_pixel, xs_pixel] = 222
                # a[ye_pixel, xe_pixel] = 222
                
                point_vector = np.array([xi - xs, yi - ys]) #vector of current point
                source_vector = np.array([xe - xs, ye - ys]) #line source vetor
                
                projection = (np.dot(point_vector, source_vector) / (np.linalg.norm(source_vector))**2) * source_vector
                projection = np.array([projection[0] + xs, projection[1] + ys])
                
                #Code for printing projection line on a
                # projection_x_pixel = int((projection[0] - xp[0] + .51 * dpx) / dpx)
                # projection_y_pixel = int((projection[1] - yp[0] + .51 * dpx) / dpy)
                #a[projection_y_pixel, projection_x_pixel] = 222
                
                d = ((xi - projection[0])**2 + (yi - projection[1])**2) ** .5
                n = int(d/h) + 1
                
                dx =  (xi - projection[0]) / (n - 1) #physical space dx
                dy = (yi - projection[1]) / (n - 1)
                
                if n == 1: #Prevents division by zero
                    continue
                
                for point_i in range(n):
                    i_x = int(np.floor( (projection[0] + point_i * dx - xp[0] + 0.51*dpx ) / dpx ) ) #pixel indices
                    i_y = int(np.floor( (projection[1] + point_i * dy - yp[0] + 0.51*dpy ) / dpy ) ) 
                    
                    if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]: # Implement mu_background here
                        a[index_y, index_x] += mu[i_y,i_x] * h
                    else : 
                        a[index_y, index_x] += mu_background * h
                p0[index_y, index_x] = mu[index_y, index_x] * (np.pi * d**2)**-1 * np.exp(-a[index_y, index_x]) 
                fluence[index_y, index_x] = np.exp(-a[index_y, index_x]) * ((np.pi * (d**2)) ** -1)
                
    return p0, a, fluence

def mu_to_p0_wedge_3d(mu, mu_background, source_start, source_end, ray_direction, theta, h, xp, yp, zp): #theta is the angle defined by the distance from the central axis of the beam, defined by ray_direction
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]
    
    xs, ys, zs = source_start
    xe, ye, ze = source_end
    
    dpx = xp[1] - xp[0]   
    dpy = yp[1] - yp[0] 
    dpz = zp[1] - zp[0]   
    
    a = np.zeros_like(mu)
    mask = np.zeros_like(mu)
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    
    zs_pixel = int((zs - zp[0] + .51 * dpz) / dpz)
    ze_pixel = int((ze - zp[0] + .51 * dpz) / dpz)
    
    for index_z in range(min(zs_pixel, ze_pixel), max(zs_pixel, ze_pixel)):
        for index_y in range(mask.shape[1]):
            for index_x in range(mask.shape[2]):
                xi = xp[index_x] #physical coordinates
                yi = yp[index_y]
                zi = zp[index_z]
                
                point_angle = np.arctan2 (yi - ys, xi - xs) - ray_direction
                
                if np.abs(point_angle) <= theta/2: 
                    mask[index_z, index_y, index_x] = 1
    
    for index_z in range(min(zs_pixel, ze_pixel), max(zs_pixel, ze_pixel)):
        for index_y in range(mask.shape[1]):
            for index_x in range(mask.shape[2]):
                if mask[index_z, index_y , index_x] == 1:
                    
                    xi = xp[index_x]
                    yi = yp[index_y]
                    zi = zp[index_z]
                    
                    source_z = np.array([xs, ys, zi])
                    d = ((xi - source_z[0])**2 + (yi - source_z[1])**2)**0.5
                    
                    n = int(d / h) + 1
                    
                    dx = (xi - source_z[0]) / (n - 1)
                    dy = (yi - source_z[1]) / (n - 1)
                    dz = (zi - source_z[2]) / (n - 1)
                    
                    for point_i in range(n):
                        i_x = int(np.floor((source_z[0] + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
                        i_y = int(np.floor((source_z[1] + point_i * dy - yp[0] + 0.51 * dpy) / dpy))
                        #i_z = int(np.floor((source_z[2] + point_i * dz - zp[0] + 0.51 * dpz) / dpz))

                        if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1]: #and 0 <= i_z < mu.shape[0]:
                            a[index_z, index_y, index_x] += mu[index_z, i_y, i_x] * h
                        else : 
                            a[index_z, index_y, index_x] += mu_background * h
                        
                    p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)
                    fluence[index_z, index_y, index_x] = np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1)
                
    return p0, a, fluence

def mu_to_p0_wedge_variable_beam_3d(mu, mu_background, source_start, source_end, ray_direction, theta, h, xp, yp, zp, I): #theta is the angle defined by the distance from the central axis of the beam, defined by ray_direction
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]
    
    xs, ys, zs = source_start
    xe, ye, ze = source_end
    
    dpx = xp[1] - xp[0]   
    dpy = yp[1] - yp[0] 
    dpz = zp[1] - zp[0]   
    
    a = np.zeros_like(mu)
    mask = np.zeros_like(mu)
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    
    zs_pixel = int((zs - zp[0] + .51 * dpz) / dpz)
    ze_pixel = int((ze - zp[0] + .51 * dpz) / dpz)
    
    for index_z in range(min(zs_pixel, ze_pixel), max(zs_pixel, ze_pixel)):
        
        theta_start = ray_direction - (theta/2)
        theta_end = ray_direction + (theta/2)
        d_theta = theta / (len(I[index_z]) -1)
        
        for index_y in range(mask.shape[1]):
            for index_x in range(mask.shape[2]):
                xi = xp[index_x] #physical coordinates
                yi = yp[index_y]
                zi = zp[index_z]
                
                point_angle = np.arctan2 (yi - ys, xi - xs)
                
                
                # #Searching for the clockwise, counter-clockwise dictionary entries.
                if min(theta_end, theta_start) <= point_angle <= max(theta_end, theta_start):
                    I_section = I[index_z]
                    clockwise_ray_i = int(np.ceil((point_angle - theta_start) / d_theta))
                    countercw_ray_i = int(np.floor((point_angle - theta_start) / d_theta))
                    
                    clockwise_ray_theta = theta_start + (clockwise_ray_i * d_theta)
                    countercw_ray_theta =  theta_start + (countercw_ray_i * d_theta)
                    
                    clockwise_ray_Intensity = I_section[clockwise_ray_i]
                    countercw_ray_Intensity = I_section[countercw_ray_i]
                        
                    ray_intensity = ((np.abs(point_angle - countercw_ray_theta) / d_theta) * clockwise_ray_Intensity) + ((np.abs(point_angle - clockwise_ray_theta) / d_theta) * countercw_ray_Intensity)
                    
                    source_z = np.array([xs, ys, zi])
                    d = ((xi - source_z[0])**2 + (yi - source_z[1])**2)**0.5
                    
                    n = int(d / h) + 1
                    
                    dx = (xi - source_z[0]) / (n - 1)
                    dy = (yi - source_z[1]) / (n - 1)
                    dz = (zi - source_z[2]) / (n - 1)
                    
                    for point_i in range(n):
                        i_x = int(np.floor((source_z[0] + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
                        i_y = int(np.floor((source_z[1] + point_i * dy - yp[0] + 0.51 * dpy) / dpy))

                        if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1]: 
                            a[index_z, index_y, index_x] += mu[index_z, i_y, i_x] * h
                        else : 
                            a[index_z, index_y, index_x] += mu_background * h
                        
                    p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (d**3)) ** -1) * ray_intensity
                    fluence[index_z, index_y, index_x] = np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (d**3)) ** -1) * ray_intensity

    return p0, a, fluence


def mu_to_p0_wedge_variable_beam(mu, mu_background, source, h, xp: np.array, yp: np.array, theta, direction, I): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
    xs, ys = source                                                                 #theta is the width of the "flashlight", direction is the "pointed" direction of the flashlight
    assert mu.shape[1]==xp.shape[0]
    assert mu.shape[0]==yp.shape[0]
    
    dpx  = xp[1] - xp[0] #spacing between sampling points
    dpy = yp[1] - yp[0]
    
    theta_start = direction - (theta/2)
    theta_end = direction + (theta/2)
    d_theta = theta / (len(I) -1)
    
    a = np.zeros_like(mu)
    mask =  np.zeros_like(mu)
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    
    ray_intensity = 0
    for index_y in range(mask.shape[0]):
        for index_x in range(mask.shape[1]):
            xi = xp[index_x] #physical coordinates
            yi = yp[index_y]
            point_angle = np.arctan2 (yi - ys, xi - xs)# - direction
            
            
            if theta_start <= point_angle <= theta_end:
                clockwise_ray_i = int(np.ceil((point_angle - theta_start) / d_theta))
                countercw_ray_i = int(np.floor((point_angle - theta_start) / d_theta))
                
                clockwise_ray_theta = theta_start + (clockwise_ray_i * d_theta)
                countercw_ray_theta = theta_start + (countercw_ray_i * d_theta)
                
                clockwise_ray_Intensity = I[clockwise_ray_i]
                countercw_ray_Intensity = I[countercw_ray_i]
                
                ray_intensity = (
                    (abs(point_angle - countercw_ray_theta) / d_theta) * clockwise_ray_Intensity
                    + (abs(point_angle - clockwise_ray_theta) / d_theta) * countercw_ray_Intensity
                )
                
                d = ((xi-xs)**2 + (yi-ys)**2) **0.5 #euclidean distance between source and target
                n = int(d/h) + 1 # of discrete point

                dx =  (xi - xs) / (n - 1) #physical space dx
                dy = (yi - ys) / (n - 1)
                
                for point_i in range(n):
                    i_x = int(np.floor( (xs + point_i * dx - xp[0] + 0.51*dpx ) / dpx ) ) #pixel indices
                    i_y = int(np.floor( (ys + point_i * dy - yp[0] + 0.51*dpy ) / dpy ) ) 
                    
                    if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]:
                        #a[index_x, index_x] = .0
                        a[index_y, index_x] += mu[i_y,i_x] * h
                    
                    else: #mu_background
                        a[index_y, index_x] += mu_background * h
                        
                p0[index_y, index_x] = mu[index_y, index_x] * np.exp(-a[index_y, index_x]) * ((np.pi * (d**2)) ** -1) * ray_intensity
                fluence[index_y, index_x] = np.exp(-a[index_y, index_x]) * ((np.pi * (d**2)) ** -1) * ray_intensity
                
    return p0, a, fluence

def mu_to_p0_cone_variable_beam_3d(mu, mu_background, source, h, xp: np.array, yp: np.array, zp: np.array, direction_vector, theta, I):
    xs, ys, zs = source
    
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]
    
    dpx = xp[1] - xp[0]   
    dpy = yp[1] - yp[0] 
    dpz = zp[1] - zp[0]   
    
    d_theta = theta / (len(I) -1)
    
    a = np.zeros_like(mu)
    p0 = np.zeros_like(mu)
    fluence = np.zeros_like(mu)
    mask = np.zeros_like(mu)
    for index_z in range(mask.shape[0]):
        for index_y in range(mask.shape[1]):
            for index_x in range(mask.shape[2]):
                xi = xp[index_x]
                yi = yp[index_y]
                zi = zp[index_z]
                
                vector_shift = np.array([xi - xs, yi - ys, zi - zs])
                point_angle = np.arccos(   np.dot(direction_vector, vector_shift)  /  (np.linalg.norm(direction_vector) * np.linalg.norm(vector_shift)   )   )
                
                if point_angle <= theta / 2:
                    mask[index_z, index_y, index_x] = 1
                    
                    outside_ray_i = int(np.ceil((point_angle) / d_theta))
                    inside_ray_i = int(np.floor((point_angle) / d_theta))
                    
                    outside_ray_theta = outside_ray_i * d_theta
                    inside_ray_theta = inside_ray_i * d_theta
                    
                    outside_ray_Intensity = I[outside_ray_i]
                    inside_ray_Intensity = I[inside_ray_i]
                    
                    ray_intensity = (
                        (abs(point_angle - inside_ray_theta) / d_theta) * outside_ray_Intensity
                        + (abs(point_angle - outside_ray_theta) / d_theta) * inside_ray_Intensity
                    )
                    
                    
                    d = ((xi - xs)**2 + (yi - ys)**2 + (zi - zs)**2)**0.5
                    n = int(d / h) + 1
                    
                    dx = (xi - xs) / (n - 1)
                    dy = (yi - ys) / (n - 1)
                    dz = (zi - zs) / (n - 1)
                    
                    for point_i in range(n):
                        i_x = int(np.floor((xs + point_i * dx - xp[0] + 0.51 * dpx) / dpx))
                        i_y = int(np.floor((ys + point_i * dy - yp[0] + 0.51 * dpy) / dpy))
                        i_z = int(np.floor((zs + point_i * dz - zp[0] + 0.51 * dpz) / dpz))
                        
                        if 0 <= i_x < mu.shape[2] and 0 <= i_y < mu.shape[1] and 0 <= i_z < mu.shape[0]:
                            a[index_z, index_y, index_x] += mu[i_z, i_y, i_x] * h
                        else: 
                            a[index_z, index_y, index_x] += mu_background * h
                            
                    p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1) * ray_intensity
                    fluence[index_z, index_y, index_x] = np.exp(-a[index_z, index_y, index_x]) * ((np.pi * (4/3) * (d**3)) ** -1) * ray_intensity
            
    return p0, a, fluence