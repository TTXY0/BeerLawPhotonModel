import numpy as np
# from numba import jit, cuda

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

            # if np.cross(source_direction_vector, sample_direction_vector)[2] > 0: 
            #     orthogonal_direction_vector = (source_direction_vector[1], -source_direction_vector[0])
            # elif np.cross(source_direction_vector, sample_direction_vector)[2] < 0:
            #     orthogonal_direction_vector = (-source_direction_vector[1], source_direction_vector[0])
            
def mu_to_p0_line(mu, source_start, source_end, ray_direction, h, xp, yp):
    assert mu.shape[1] == xp.shape[0]
    assert mu.shape[0] == yp.shape[0]
    
    xs, ys = source_start
    xe, ye = source_end
    # xe = xs + source_length * np.cos(source_direction)
    # ye = ys + source_length * np.sin(source_direction)
    #print(xe -xs, ye-ys)
    p_source_vector = np.array([xe - xs, ye - xs]) #physical source vector
    #print(p_source_vector)
    
    dpx = xp[1] - xp[0]  # spacing between sampling points
    dpy = yp[1] - yp[0]
    
    
    a = np.zeros_like(mu)
    
    
    xs_pixel = int( np.floor((xs - xp[0]) / dpx) )
    ys_pixel = int( np.floor((ys - yp[0]) / dpy) )
    xe_pixel = int( np.floor((xe - xp[0]) / dpx) )
    ye_pixel = int( np.floor((ye - yp[0]) / dpy) )
    
    
    
    mask = np.zeros_like(mu)
    #print(xs_pixel, ys_pixel, xe_pixel, ye_pixel)
    mask[ys_pixel, xs_pixel] = 1
    mask[ye_pixel, xe_pixel] = 1
    
    for index_y in range(mask.shape[0]):
        for index_x in range(mask.shape[1]):
            # index_y = 50
            # index_x = 30

            source_vector = np.array([xe_pixel - xs_pixel, ye_pixel - ys_pixel])
            point_vector = np.array([index_x - xs_pixel, index_y - ys_pixel])
            #(y-b)/m = x
            perp_slope = -(source_vector[0] / source_vector[1])
            #b = -mx1 + y1
            b_start = -perp_slope * xs_pixel + ys_pixel
            b_end = -perp_slope * xe_pixel + ye_pixel
            
            x_start = (index_y - b_start) / perp_slope
            x_end = (index_y - b_end) / perp_slope
            

            projection = (np.dot(point_vector, source_vector) / (np.linalg.norm(source_vector) ** 2) ) * source_vector
            projection = projection + np.array([xs_pixel, ys_pixel]) 
            
            
            
            unit_vector = np.array([index_x - projection[0], index_y - projection[1]])
            if np.allclose(unit_vector, np.array([0,0])):
                continue
            else:
                unit_vector = (unit_vector / np.linalg.norm(unit_vector))
            #print("skjflakjsjkla",np.dot(point_vector / np.linalg.norm(point_vector), p_source_vector / np.linalg.norm(p_source_vector)))
            #print((np.dot(point_vector/np.linalg.norm(point_vector), source_vector/np.linalg.norm(source_vector))))
            if not np.allclose(unit_vector, ray_direction, rtol = .2) or not np.min([x_end, x_start]) <= projection[0] <= np.max([x_start, x_end]):# or np.array_equal(np.dot(point_vector, p_source_vector), np.array([0,0])):
                #this is to filter out the points matches that do not math ray-direction, points not in between the bounding box of the "line" light source,
                #and that the point is not parallel to the light source, which will throw an error below
                #print(np.allclose(unit_vector, ray_direction))
                #print(unit_vector, ray_direction)
                #print(not np.array_equal(unit_vector, ray_direction) or not (np.min([xs_pixel, xe_pixel]) <= projection[0] <= np.max([xe_pixel, xs_pixel]) and np.min([ys_pixel, ye_pixel]) <= projection[1] <= np.max([ye_pixel, ys_pixel])))
                continue
            else: 
                #print("true")
                mask[index_y,index_x] = 1
                
    p0 = np.zeros_like(mu)
    for index_y in range(mu.shape[0]):
            for index_x in range(mu.shape[1]):
                if mask[index_y , index_x] == 1:
                    
                    xi = xp[index_x] #physical coordinates
                    yi = yp[index_y]
                    
                    p_point_vector = np.array([xi - xs, yi - ys]) #point vector relative to the starting point of the line source

                    #print(np.dot(point_vector, p_source_vector))
  
                    #if not np.allclose(np.dot(p_point_vector, p_source_vector), np.array([0,0]), rtol=.2): #if the point vectors is not vertical
                        #print(point_vector, source_vector)
                        #print("dpt product", np.dot(source_vector, p_source_vector))
                        
                    
                    #projection = (np.dot(p_source_vector, p_point_vector) / (np.linalg.norm(p_point_vector) ** 2)) * p_point_vector
                    projection = (np.dot(p_point_vector, p_source_vector) / (np.linalg.norm(p_source_vector) ** 2)) * p_source_vector
                    #projection = projection + np.array([ys, xs])
                    source = projection #the projection becomes a "source" on the line
                    print(source)
                    # else: 
                    #     #print(p_source_vector)
                    #     source = np.array([xs, (index_y * dpy) + yp[0]])
                        
                    xs_point = source[0]
                    ys_point = source[1]
                    
                    source_ix = (int(np.floor( (xs_point - xp[0] + .51*dpx) / dpx ) ))
                    source_iy = (int(np.floor( (ys_point - yp[0] + .51 *dpy) / dpy ) ))
                    
                    print(source_iy, source_ix)
                    if 0<= source_iy < mu.shape[0] and 0<= source_ix < mu.shape[1]:
                        a[source_iy, source_ix] = 111
                    
                    # print(source[0])
                    # (print(xi, yi, xs, ys))
                    d = ((xi-xs_point)**2 + (yi-ys_point)**2) **0.5 #euclidean distance between source and target
                    # print(d, h)
                    #print("d,h: ", d, h)
                    n = int(d/h) + 1 # of discrete point
                    #print("n:", n)

                    if n != 1:
                        dx =  (xi - xs_point) / (n - 1) #physical space dx
                        dy = (yi - ys_point) / (n - 1)
                    else: 
                        dx = xi-xs_point
                        dy = yi-ys_point
                        
                    
                        
                    
                    # for point_i in range(n):
                    #     # print(xs_point, dpx)
                    #     # print((xs_point + point_i * dx - xp[0] + 0.51*dpx ) / dpx )
                    #     i_x = int(np.floor( (xs_point + point_i * dx - xp[0] + 0.51*dpx ) / dpx ) ) #pixel indices
                    #     i_y = int(np.floor( (ys_point + point_i * dy - yp[0] + 0.51*dpy ) / dpy ) ) 
                        
                    #     if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]:
                    #         #a[index_x, index_x] = .0
                    #         a[index_y, index_x] += mu[i_y,i_x] * h
                            
                    # p0[index_y, index_x] = mu[index_y, index_x] * np.exp(-a[index_y, index_x])

    return p0, a, mask
            

