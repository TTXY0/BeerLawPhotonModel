import numpy as np
def mu_to_p0(mu, source, h, xp: np.array, yp: np.array): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
    xs, ys = source
    assert mu.shape[1]==xp.shape[0]
    assert mu.shape[0]==yp.shape[0]
    
    dpx  = xp[1]-xp[0] #spacing between sampling points
    dpy = yp[1]-yp[0]
    a = np.zeros_like(mu)

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
                 
    return mu * np.exp(-a), a 


def mu_to_p0_cone(mu, source, h, xp: np.array, yp: np.array, theta, direction): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
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
                        
                p0[index_y, index_x] = mu[index_y, index_x] * np.exp(-a[index_y, index_x])
            
    return p0, a, mask


def mu_to_p0_3d(mu, source, h, xp, yp, zp):
    xs, ys, zs = source
    assert mu.shape[2] == xp.shape[0]
    assert mu.shape[1] == yp.shape[0]
    assert mu.shape[0] == zp.shape[0]
    
    dpx = xp[1] - xp[0]
    dpy = yp[1] - yp[0]
    dpz = zp[1] - zp[0]
    
    a = np.zeros_like(mu)
    
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
                    
    return mu * np.exp(-a), a

def mu_to_p0_cone_3d(mu, source, h, xp: np.array, yp: np.array, zp: np.array, direction_vector, theta):
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
                        
                    p0[index_z, index_y, index_x] = mu[index_z, index_y, index_x] * np.exp(-a[index_z, index_y, index_x])
            
    return p0, a, mask


