#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import numpy as np
from scipy import ndimage


import espacios_color as espc




def super_fft2D(imagen, convertir_YIG = False):
    
    imagen_use = np.copy(imagen)
    
    if imagen_use.max() > 1.0:
        imagen_use = imagen_use/255.0
    
    # Me quedo solo con la luminancia
    # Paso a luminancia si no esta ya en ese formato
    if convertir_YIG:
        imagen_use = espc.rgb2yiq(imagen_use)    
    
    luminancia = imagen_use[:,:,0]
    
    # Reservo memoria
    dims_img = np.squeeze(luminancia).shape
    fft_aux = np.zeros(dims_img,dtype=np.complex_)
    fft_aux_2 = np.zeros(dims_img,dtype=np.complex_)
    fft_image = np.zeros(dims_img,dtype=np.complex_)
    
    # Transformo por filas
    for idx_y in range(dims_img[1]):
        fft_aux[:,idx_y] = np.fft.fft(luminancia[:,idx_y])
        
    # Transformo por columnas
    for idx_x in range(dims_img[0]):
        fft_aux_2[idx_x,:] = np.fft.fft(fft_aux[idx_x,:])
        
    # Acomodo los cuadrantes
    cuadrante_0 = fft_aux_2[:int(dims_img[0]/2),:int(dims_img[1]/2)]
    cuadrante_1 = fft_aux_2[int(dims_img[0]/2):,:int(dims_img[1]/2)]
    cuadrante_2 = fft_aux_2[:int(dims_img[0]/2),int(dims_img[1]/2):]
    cuadrante_3 = fft_aux_2[int(dims_img[0]/2):,int(dims_img[1]/2):]
    
    fft_image[:int(dims_img[0]/2),:int(dims_img[1]/2)] = cuadrante_3
    fft_image[int(dims_img[0]/2):,int(dims_img[1]/2):] = cuadrante_0
    fft_image[int(dims_img[0]/2):,:int(dims_img[1]/2)] = cuadrante_2
    fft_image[:int(dims_img[0]/2),int(dims_img[1]/2):] = cuadrante_1
    
    
        
    modulo = 23*np.log10(np.sqrt(fft_image.imag**2 + fft_image.real**2)+1e-66)
    fase = np.angle(fft_image)
        
    return modulo, fase
        
    
def super_invfft2D(fft_modulo_in, ft_fase_in):
    
    fft_modulo_use = np.copy(fft_modulo_in)
    ft_fase_use = np.copy(ft_fase_in)
    
    # Paso a un vector complejo
    fft_modulo_use = 10**(fft_modulo_use/23)
    fft_real = fft_modulo_use*np.cos(ft_fase_use)
    fft_imag = fft_modulo_use*np.sin(ft_fase_use)
    
    fft_use = fft_real + 1j*fft_imag
    

    # Reservo memoria
    dims_img = fft_modulo_use.shape
    fft_acom = np.zeros(dims_img,dtype=np.complex_)
    ifft_aux = np.zeros(dims_img,dtype=np.complex_)
    imagen_out = np.zeros(dims_img,dtype=np.complex_)
    
    # Re-Acomodo los cuadrantes
    cuadrante_0 = fft_use[:int(dims_img[0]/2),:int(dims_img[1]/2)]
    cuadrante_1 = fft_use[int(dims_img[0]/2):,:int(dims_img[1]/2)]
    cuadrante_2 = fft_use[:int(dims_img[0]/2),int(dims_img[1]/2):]
    cuadrante_3 = fft_use[int(dims_img[0]/2):,int(dims_img[1]/2):]
    
    fft_acom[:int(dims_img[0]/2),:int(dims_img[1]/2)] = cuadrante_3
    fft_acom[int(dims_img[0]/2):,int(dims_img[1]/2):] = cuadrante_0
    fft_acom[int(dims_img[0]/2):,:int(dims_img[1]/2)] = cuadrante_2
    fft_acom[:int(dims_img[0]/2),int(dims_img[1]/2):] = cuadrante_1
    #fft_acom = fft_use

    # Des-Transformo por columnas
    for idx_x in range(dims_img[0]):
        ifft_aux[idx_x,:] = np.fft.ifft(fft_acom[idx_x,:])
    
    # Des-Transformo por filas
    for idx_y in range(dims_img[1]):
        imagen_out[:,idx_y] = np.fft.ifft(ifft_aux[:,idx_y])
        
        
    # Conservo solo la parte real, en el resto solo hay ruido numerico
    imagen_out = imagen_out.real
                
    return imagen_out
    
    
    