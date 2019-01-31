#!/usr/bin/python
# -*- coding: iso-8859-15 -*-



import numpy as np
from scipy import ndimage

LIM_I = 0.59590059
LIM_Q = 0.52273617

RGB2YIQ = np.array([[0.299,      0.587,        0.114],  [0.59590059, -0.27455667, -0.32134392],  [0.21153661, -0.52273617, 0.31119955]])


def check_LUMINANCIA(imagen_LUMINANCIA, modo = 'int'):
    
    if modo == 'int':
        VAL_MAX = 255
    elif modo == 'float':
        VAL_MAX = 1
    else:
        print('Error modo no valido')
        return
      
    
    image_shape = imagen_LUMINANCIA.shape
    num_px = image_shape[0]*image_shape[1]

    lista_fix = np.argwhere((imagen_LUMINANCIA[:,:] > VAL_MAX ) |  
                            (imagen_LUMINANCIA[:,:] < 0.0 ) )
    for px in lista_fix:
        idx_x = px[0]
        idx_y = px[1]
        if imagen_LUMINANCIA[idx_x,idx_y] < 0:
            imagen_LUMINANCIA[idx_x,idx_y] = 0
        elif imagen_LUMINANCIA[idx_x,idx_y] > VAL_MAX:
            imagen_LUMINANCIA[idx_x,idx_y] = VAL_MAX

    return imagen_LUMINANCIA

def check_RGB(imagen_RGB, modo = 'int'):
    
    if modo == 'int':
        VAL_MAX = 255
    elif modo == 'float':
        VAL_MAX = 1
    else:
        print('Error modo no valido')
        return
      
    
    image_shape = imagen_RGB.shape
    num_px = image_shape[0]*image_shape[1]

    lista_fix = np.argwhere((imagen_RGB[:,:,0] > VAL_MAX ) | 
                            (imagen_RGB[:,:,1] > VAL_MAX ) | 
                            (imagen_RGB[:,:,2] > VAL_MAX ) | 
                            (imagen_RGB[:,:,0] < 0.0 ) | 
                            (imagen_RGB[:,:,1] < 0.0 ) | 
                            (imagen_RGB[:,:,2] < 0.0 ) )
    for px in lista_fix:
        idx_x = px[0]
        idx_y = px[1]
        for i in range(3):
            if imagen_RGB[idx_x,idx_y,i] < 0:
                imagen_RGB[idx_x,idx_y,i] = 0
            elif imagen_RGB[idx_x,idx_y,i] > VAL_MAX:
                imagen_RGB[idx_x,idx_y,i] = VAL_MAX


    return imagen_RGB


def check_YIQ(imagen):

    image_shape = imagen.shape
    num_px = image_shape[0]*image_shape[1]

    lista_fix = np.argwhere((imagen[:,:,0]         >= 1.0                  ) | 
                            (np.abs(imagen[:,:,1]) >= np.abs(RGB2YIQ[1,0]) ) | 
                            (np.abs(imagen[:,:,2]) >= np.abs(RGB2YIQ[2,1]) )  )

    for px in lista_fix:
        idx_x = px[0]
        idx_y = px[1]
        # Y
        if imagen[idx_x,idx_y,0] < 0:
            imagen[idx_x,idx_y,0] = 0
        elif imagen[idx_x,idx_y,0] > 1:
            imagen[idx_x,idx_y,0] = 1
        # I 
        if imagen[idx_x,idx_y,1] < -LIM_I:
            imagen[idx_x,idx_y,1] = -LIM_I
        elif imagen[idx_x,idx_y,1] > LIM_I:
            imagen[idx_x,idx_y,1] = LIM_I
        # Q 
        if imagen[idx_x,idx_y,2] < -LIM_Q:
            imagen[idx_x,idx_y,2] = -LIM_Q
        elif imagen[idx_x,idx_y,2] > LIM_Q:
            imagen[idx_x,idx_y,2] = LIM_Q
      

    return imagen


def check_YIQ_transform(imagen, imagen_old):

    image_shape = imagen.shape
    num_px = image_shape[0]*image_shape[1]

    lista_fix = np.argwhere((imagen[:,:,0]         >= 1.0                  ) | 
                            (np.abs(imagen[:,:,1]) >= np.abs(RGB2YIQ[1,0]) ) | 
                            (np.abs(imagen[:,:,2]) >= np.abs(RGB2YIQ[2,1]) )  )

    for px in lista_fix:
        idx_x = px[0]
        idx_y = px[1]
        # obtengo la recta 
        P = imagen[idx_x,idx_y,:] - imagen_old[idx_x,idx_y,:]
        # Resuelvo para el punto que primero sature
        a = np.array([0,0,0,0,0], dtype=np.float32)
        a[0] =           (1.0-imagen_old[idx_x,idx_y,0]) / P[0]
        a[1] = ( RGB2YIQ[1,0]-imagen_old[idx_x,idx_y,1]) / P[1]
        a[2] = (-RGB2YIQ[1,0]-imagen_old[idx_x,idx_y,1]) / P[1]
        a[3] = ( RGB2YIQ[2,1]-imagen_old[idx_x,idx_y,2]) / P[2]
        a[4] = (-RGB2YIQ[2,1]-imagen_old[idx_x,idx_y,2]) / P[2]
        a[np.argwhere(np.isnan(a))] = 9e9
        idx_a_use = np.argmin(np.abs(a))
        # Obtengo el valor saturado del pixel
        imagen[idx_x,idx_y,:] = P*a[idx_a_use] + imagen_old[idx_x,idx_y,:]
        #print(imagen[idx_x,idx_y,:])

    return imagen



def rgb2gray(imagen):
       
    imagen_use = rgb2yiq(imagen)
    
    imagen_out = imagen_use[:,:,0]
        
    return imagen_out




def rgb2yiq(imagen):
    # Las componentes RGB en la matriz ndimensional que representa 
    # la imagen parecen estar puestas como dimension fila cuando
    # numpy hace la multiplicación, así que tenemos que 
    # transponer la matriz de conversión
    imagen_use = np.double(imagen)
    if imagen.max() > 1.0:
        imagen_use = imagen_use/255.0
        
    imagen_out = np.matmul(imagen_use, RGB2YIQ.T)

    return imagen_out


def yiq2rgb(imagen):

    imagen_out = np.matmul(imagen, np.linalg.inv(RGB2YIQ.T))

    # arreglo pixeles negativos o altos
    imagen_out = (imagen_out*255.0)
    imagen_out = check_RGB(imagen_out).astype(np.uint8)

    return (imagen_out)


def aplicar_alpha(imagen, alfa):

    imagen_out = np.copy(imagen)

    imagen_out[:,:,0] = imagen[:,:,0]*alfa

    #imagen_out = check_YIQ_transform(imagen_out,imagen)
    imagen_out = check_YIQ(imagen_out)

    return imagen_out


def aplicar_beta(imagen, beta):

    imagen_out = np.copy(imagen)

    imagen_out[:,:,1] = imagen[:,:,1]*beta
    imagen_out[:,:,2] = imagen[:,:,2]*beta

    #imagen_out = check_YIQ_transform(imagen_out,imagen)
    imagen_out = check_YIQ(imagen_out)

    return imagen_out

