#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import numpy as np
from scipy import ndimage


import espacios_color as espc


def histogramear(imagen_YIQ, bines, normalizar = False):
    
    # Obtengo el vector de valores de luminnacia
    vector2hist = imagen_YIQ[:,:,0].flatten()
    
    if vector2hist.max() > 1.0:
        vector2hist = vector2hist/255.0
    
    # Creo los bines a utilizar
    bin_edges = np.linspace(0,1,bines)
    
    hist, bin_edges_return = np.histogram(vector2hist, bin_edges)
    
    # Normalizo si se pide
    if normalizar:
        hist = hist / hist.max()
        
    
    # Paso de bordes a centros de bins, para poder plotear
    bin_edges_return = bin_edges_return + (bin_edges[1]-bin_edges[0])*0.5
    
    return hist, bin_edges_return[:-1]

def transformar_luminancia(imagen_YIQ, thrs_min, thrs_max, pend_rescale, offset_rescale, offset_base = 0):
    
    imagen_use = np.copy(imagen_YIQ)
    imagen_use_Y = np.copy(imagen_use[:,:,0])
    
    imagen_use[:,:,0] = np.multiply(imagen_use[:,:,0],pend_rescale) + offset_rescale
    imagen_use[:,:,0][imagen_use_Y > thrs_max] = 1
    imagen_use[:,:,0][imagen_use_Y < thrs_min] = 0
    
    if offset_base != 0:
        imagen_use[:,:,0] = imagen_use[:,:,0] + offset_base
        imagen_use = espc.check_YIQ(imagen_use)
    
    return imagen_use
    

def maxim_rango_dinamico(imagen_YIQ, threshold = 0.05, bines = 256):

    # Hago un histograma
    hist, bines = histogramear(imagen_YIQ, bines, normalizar = True)
        
    # Busco los limites donde el porcentaje de pixels es menor al 
    # limite definido.
    hist[-1]=0 # Elimino los saturados
    bines_aux = bines[hist > threshold]
    bin_min = bines_aux[0]
    bin_max = bines_aux[-1]

    # Armo una recta con los limites dados
    # 0 = bin_min * a + b
    # -b = bin_min*a
    # 1 = bin_max * a + b
    # (1-b)/a = bin_max
    # (1+bin_min*a)/a = bin_max = (1/a) + bin_min 
    # 1/a = bin_max - bin_min
    # a = 1/(bin_max - bin_min)
    # b = -bin_min/(bin_max - bin_min)
    # Los valores y mayores a los limites los saturo
    b = -bin_min/(bin_max - bin_min)
    a = 1/(bin_max - bin_min)

    
    # Transformo y retorno
    return transformar_luminancia(imagen_YIQ, bin_min, bin_max, a, b)
    

    
    
def transform_sqrt(imagen_YIQ):
    
    imagen_use = np.copy(imagen_YIQ)
    
    # Obtengo el vector de valores de luminnacia
    ilumin = imagen_use[:,:,0]
    
    if ilumin.max() > 1.0:
        ilumin = ilumin/255.0
        
    imagen_use[:,:,0] = np.sqrt(ilumin)
    
    return imagen_use
    
def transform_pow(imagen_YIQ):
    
    imagen_use = np.copy(imagen_YIQ)
    
    # Obtengo el vector de valores de luminnacia
    ilumin = imagen_use[:,:,0]
    
    if ilumin.max() > 1.0:
        ilumin = ilumin/255.0
        
    imagen_use[:,:,0] = ilumin**2
    
    return imagen_use
    
    
    