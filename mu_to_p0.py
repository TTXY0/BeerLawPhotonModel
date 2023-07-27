import numpy as np
def mu_to_p0(mu, source, h, xp: np.array, yp: np.array): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
    xs, ys = source
    dpx  = xp[1]-xp[0] #spacing between sampling points
    dpy = yp[1]-yp[0]
    P = np.zeros_like(mu)
    
    pixelsx = []
    pixelsy = []
    
    for index_y in range(mu.shape[0]):
        for index_x in range(mu.shape[1]):
            xi = xp[index_x] #physical coordinates
            yi = yp[index_y]
            
            d = ((xi-xs)**2 + (yi-ys)**2) **0.5 #euclidean distance between source and target
            n = int(d/h) + 1 # of discrete point

            dx =  (xi - xs) / (n - 1) #physical space dx
            dy = (yi - ys) / (n - 1)
            
            for point_i in range(n):
                pixelsx.append((xs + (point_i * dx) - xp[0])/ (dpx) )#pixel indices
                pixelsy.append((ys + (point_i * dy) - yp[0])/ (dpy))
                
                
                i_x = int(np.floor(xs + (point_i * dx) - xp[0])/ (dpx)) #pixel indices
                i_y = int(np.floor(ys + (point_i * dy) - yp[0])/ (dpy)) 
                if 0 <= i_x < mu.shape[0] and 0 <= i_y < mu.shape[1]:
                    P[i_y,i_x] += mu[i_y,i_x] * (h)
                
    return mu * np.exp(-P), P, pixelsx, pixelsy