import numpy as np
def mu_to_p0(mu, source, h, xp: np.array, yp: np.array): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
    xs, ys = source
    assert mu.shape[1]==xp.shape[0]
    assert mu.shape[0]==yp.shape[0]
    
    dpx  = xp[1]-xp[0] #spacing between sampling points
    dpy = yp[1]-yp[0]
    a = np.zeros_like(mu)
    
    # pixelsx = {}
    # pixelsy = {}

    for index_y in range(mu.shape[0]):
        for index_x in range(mu.shape[1]):
            xi = xp[index_x] #physical coordinates
            yi = yp[index_y]
            
            
            d = ((xi-xs)**2 + (yi-ys)**2) **0.5 #euclidean distance between source and target
            n = int(d/h) + 1 # of discrete point

            dx =  (xi - xs) / (n - 1) #physical space dx
            dy = (yi - ys) / (n - 1)
            
            # ppx = [] 
            # ppy = []
                       
            for point_i in range(n):
                # ppx.append( (xs + point_i * dx - xp[0]  ) / dpx ) #pixel indices
                # ppy.append( (ys + point_i * dy - yp[0]  ) / dpy ) 
                
                
                i_x = int(np.floor( (xs + point_i * dx - xp[0] + 0.51*dpx ) / dpx ) ) #pixel indices
                i_y = int(np.floor( (ys + point_i * dy - yp[0] + 0.51*dpy ) / dpy ) ) 
                if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]:
                    a[index_y, index_x] += mu[i_y,i_x] * h
                    
            # pixelsx[index_y, index_x] = ppx
            # pixelsy[index_y, index_x] = ppy
            
                
    return mu * np.exp(-a), a #, pixelsx, pixelsy


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