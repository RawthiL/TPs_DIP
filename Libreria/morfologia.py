#!/usr/bin/python
# -*- coding: iso-8859-15 -*-



import numpy as np
from scipy import ndimage

import espacios_color as espc
import suma_y_resta as syr
import OpsHistLum as ophl
import superFT as sft
import conv_2D as C2D

def min_math(imagen, kernel = 0):  
    return imagen.min()

def max_math(imagen, kernel = 0):  
    return imagen.max()

def median_math(imagen, kernel = 0):  
    return np.median(imagen)

def erosion(imagen, vecindad):
    
    kernel_dummy = np.ones((vecindad,vecindad))
    
    return C2D.conv_2D(imagen, kernel_dummy, kernel_function = min_math)
    
            
def dilatacion(imagen, vecindad):
    
    kernel_dummy = np.ones((vecindad,vecindad))
    
    return C2D.conv_2D(imagen, kernel_dummy, kernel_function = max_math)

def mediana(imagen, vecindad):
        
    kernel_dummy = np.ones((vecindad,vecindad))
    
    return C2D.conv_2D(imagen, kernel_dummy, kernel_function = median_math)

def apertura(imagen, vecindad):
    imag_out = erosion(imagen, vecindad)
    return dilatacion(imag_out, vecindad)

def clausura(imagen, vecindad):
    imag_out = dilatacion(imagen, vecindad)
    return erosion(imag_out, vecindad)

def borde_interior(imagen, vecindad):
    imag_out = erosion(imagen, vecindad)
    return imag_out - imagen

def borde_exterior(imagen, vecindad):
    imag_out = dilatacion(imagen, vecindad)
    return imag_out - imagen

def arriba_sombrero(imagen, vecindad):
    imag_out = apertura(imagen, vecindad)
    return imagen - imag_out