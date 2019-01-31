#!/usr/bin/python
# -*- coding: iso-8859-15 -*-



import numpy as np
from scipy import ndimage

import espacios_color as espc
import suma_y_resta as syr
import OpsHistLum as ophl
import superFT as sft


def gen_kernel_plano(tamanio):
    out_mat = np.ones((tamanio,tamanio))
    return out_mat/out_mat.sum()

def gen_kernel_Bartlett(tamanio):
    
    vector = np.zeros(tamanio)
    cuenta = 1
    i = 0
    while i < (np.ceil(tamanio/2).astype(np.int32)):
        vector[i] = cuenta
        vector[-i-1] = cuenta
        cuenta = cuenta + 1
        i = i + 1
    vector[i] = cuenta
    
    vector = np.expand_dims(vector,axis = 0)
    out_mat = np.matmul(vector.T,vector)

    return out_mat/out_mat.sum()

def gen_kernel_Gauss(tamanio):
    
    vector = np.zeros(tamanio)
    vector_aux = np.zeros(tamanio)
    vector[0] = 1
    
    for i in range(tamanio-1):
        vector_aux[0] = 1
        j = 1
        while vector[j] != 0:
            vector_aux[j] = vector[j-1]+vector[j]
            j = j+1
        vector_aux[j] = 1
        vector = np.copy(vector_aux)
        
    vector = np.expand_dims(vector,axis = 0)
  
    
    out_mat = np.matmul(vector.T,vector)
    return out_mat/out_mat.sum()


def gen_kernel_Laplaciano(tipo):
    
    if tipo == 'v4':
        return np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    elif tipo == 'v8':
        return np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    else:
        print('Tipo de filtro no encontrado.')
        return np.nan

def gen_kernel_Sobel(orientacion):
    
    if orientacion == 'N':
        return np.array([[-1,-2,-1],
                         [ 0, 0, 0],
                         [ 1, 2, 1]])
    elif orientacion == 'S':
        return np.array([[ 1, 2, 1],
                         [ 0, 0, 0],
                         [-1,-2,-1]])
    elif orientacion == 'E':
        return np.array([[ 1, 0,-1],
                         [ 2, 0,-2],
                         [ 1, 0,-1]])
    elif orientacion == 'O':
        return np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    elif orientacion == 'NE':
        return np.array([[ 0,-1,-2],
                         [ 1, 0,-1],
                         [ 2, 1, 0]])
    elif orientacion == 'SO':
        return np.array([[ 0, 1, 2],
                         [-1, 0, 1],
                         [-2,-1, 0]])
    elif orientacion == 'NO':
        return np.array([[-2,-1, 0],
                         [-1, 0, 1],
                         [ 0, 1, 2]])
    elif orientacion == 'SE':
        return np.array([[ 2, 1, 0],
                         [ 1, 0,-1],
                         [ 0,-1,-2]])
    else:
        print('Tipo de filtro no encontrado.')
        return np.nan
    

def gen_kern_relative_idx(kernel, CHECK_SHAPE = True):
     
    (size_x, size_y) = kernel.shape

    if CHECK_SHAPE:
        if not (size_x % 2) or not (size_y % 2):
            print('Las dimenciones del kernel deben ser impares.')
            error()

    # Armo una lista
    idx_list = list()
    for i in range(size_x):
        for j in range(size_y):
            idx_list.append([-np.floor(size_x/2)+i, -np.floor(size_y/2)+j])
         
    # Paso a array de enteros
    idx_cent = np.array(idx_list, dtype=np.int32)
    # Obtengo el patron base
    idx_row = idx_cent[:,0] 
    idx_col = idx_cent[:,1] 
    
    return idx_row, idx_col

def get_padded_copy(imagen, pad_size):
    (orig_x, orig_y) = imagen.shape
    imagen_out = np.zeros( ( (orig_x + 2*pad_size), (orig_y + 2*pad_size) ) )
    
    imagen_out[pad_size:-pad_size,pad_size:-pad_size] = np.copy(imagen)
    
    for pad_x in range(pad_size):
        imagen_out[pad_x,:] = imagen_out[pad_size,:]
        imagen_out[-pad_x-1,:] = imagen_out[-pad_size-1,:]
    for pad_y in range(pad_size):
        imagen_out[:,pad_y] = imagen_out[:,pad_size]
        imagen_out[:,-pad_y-1] = imagen_out[:,-pad_size-1]
    
    return imagen_out


def std_kernel_math(imagen_pad, kernel):
    
    return np.dot(imagen_pad, kernel)

def conv_2D(imagen, kernel, kernel_function = std_kernel_math, paso_base = 1, rango_out = 1, idx_row_base = np.array([]), idx_col_base = np.array([])):
    
    if len(imagen.shape) > 2:
        print('La imagen debe tener solo un canal.')
        return
    
    # Obtengo el tamaño de la imagen
    (num_x, num_y) = imagen.shape
    (num_x_K, num_y_K) = kernel.shape
    kern_elem = kernel.size
    
    # Armo una imagen donde le adicione el margen repitiendo columnas
    pad_size = np.floor(num_x_K/2).astype(np.int32)
    imagen_pad = get_padded_copy(imagen, pad_size)
    
    # Me fijo si me pasaron la lista de indices
    if idx_row_base.size == 0:
        # Genero la lista
        (idx_row_base, idx_col_base) = gen_kern_relative_idx(kernel)
    # Obtengo indices del kernel
    idx_row_kernel = idx_row_base + pad_size
    idx_col_kernel = idx_col_base + pad_size
    

    
    # Creo imagen de salida
    imagen_out = np.zeros((int(num_x*rango_out), int(num_y*rango_out)))
    # Creo indices de imagen salida
    idx_x_out = 0
    idx_y_out = 0
    idx_y_max = 0
    # Recorro toda la imagen base
    for idx_x in np.arange(0,num_x,paso_base):
        for idx_y in np.arange(0,num_y,paso_base):
            # Obtengo los indices
            idx_row_use = idx_row_base + idx_x
            idx_col_use = idx_col_base + idx_y
            # Aplico la función, que por default es el producto punto
            imagen_out[idx_x_out:idx_x_out+rango_out,
                       idx_y_out:idx_y_out+rango_out] = kernel_function(imagen_pad[idx_row_use   ,idx_col_use   ],
                                                                        kernel    [idx_row_kernel,idx_col_kernel])
            # incremento indices
            idx_y_out = idx_y_out + rango_out
        # incremento indices
        idx_x_out = idx_x_out + rango_out
        idx_y_max = idx_y_out
        idx_y_out = 0
     
    # En modo normal la imagen queda del mismo tamaño que el de entrada
    # pero si no es asi cropeamos la imagen al tamaño final
    imagen_out = imagen_out[:idx_x_out, :idx_y_max]

    return imagen_out