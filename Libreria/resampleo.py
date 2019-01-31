#!/usr/bin/python
# -*- coding: iso-8859-15 -*-



import numpy as np
from scipy import ndimage

import espacios_color as espc
import suma_y_resta as syr
import OpsHistLum as ophl
import superFT as sft
import conv_2D as C2D

def func_upsample_hold(imagen, kernel = 1):
    
    tamanio = imagen.size
    dim_x = int(np.sqrt(tamanio))
    dim_y = int(dim_x)
    
    return np.ones((dim_x,dim_y))*imagen[0]

def func_downsample_hold(imagen, kernel = 1):
    return imagen[-1]
    
def gen_downsample_indexes(magnitud):
    
    kernel_dummy = np.ones((magnitud,magnitud))
    
    idx_row_base, idx_col_base = C2D.gen_kern_relative_idx(kernel_dummy, CHECK_SHAPE = False)
    
    central = np.argwhere((idx_row_base + idx_col_base) == 0 ) 
    
    # Pongo al central al final de la lista
    idx_row_base[central] = idx_row_base[-1]
    idx_row_base[-1] = 0
    idx_col_base[central] = idx_col_base[-1]
    idx_col_base[-1] = 0
    
    return idx_row_base, idx_col_base

def gen_upsample_indexes(magnitud):
    
    # Armo una lista
    idx_list = list()
    idx_list.append([-1,-1])
    idx_list.append([0,-1])
    idx_list.append([-1,0])
    idx_list.append([0,0])
    
         
    # Paso a array de enteros
    idx_cent = np.array(idx_list, dtype=np.int32)
    # Obtengo el patron base
    idx_row = idx_cent[:,0] 
    idx_col = idx_cent[:,1] 
    
    return idx_row, idx_col

def gen_corners_indexes(magnitud):
    
    # Armo una lista
    idx_list = list()
    idx_list.append([-magnitud,-magnitud])
    idx_list.append([0,-magnitud])
    idx_list.append([-magnitud,0])
    idx_list.append([0,0])
    
    # Paso a array de enteros
    idx_cent = np.array(idx_list, dtype=np.int32)
    # Obtengo el patron base
    idx_row = idx_cent[:,0] 
    idx_col = idx_cent[:,1] 
    
    return idx_row, idx_col
    
    


def get_bilineal_value(dist_Kernel, dist2point_x, dist2point_y, values):
    
    x1 = 0
    y1 = 0
    x2 = dist_Kernel
    y2 = dist_Kernel
    x = dist2point_x
    y = dist2point_y

    # Interpolo en X
    aux_x_y1 = ( (x2-x)/(x2-x1) * values[0] ) + ( ((x-x1)/(x2-x1)) * values[1] )
    aux_x_y2 = ( (x2-x)/(x2-x1) * values[2] ) + ( ((x-x1)/(x2-x1)) * values[3] )
    
    # Interpolo en Y
    return ( ((y2-y)/(y2-y1))*aux_x_y1 ) + ( ((y-y1)/(y2-y1))*aux_x_y2 )
    
     
def func_downsample_bilineal(imagen, kernel):
  
    # El tamaño entre pixeles lo
    # meti el valor del kernel =D
    dist_kern = np.unique(kernel)
    dist_px = dist_kern/2
    
    return get_bilineal_value(dist_kern, dist_px, dist_px, imagen)

def func_upsample_bilineal(imagen, kernel = 1):
    
    # El tamaño entre pixeles ahora lo doy vuelta
    # es decir, upsampleo diciendo que hay mas 
    # espacio entre pixeles
    dist_kern = np.unique(kernel)
    
    # Tengo varios puntos a samplear ahora, dados
    # por el tamaño del upsample
    out_vals = np.zeros((int(dist_kern),int(dist_kern)))
    
    # Las distancias van a estar moduladas segun
    # el pixel a samplear
    for idx_x in range(int(dist_kern)):
        for idx_y in range(int(dist_kern)):
            out_vals[idx_x,idx_y] = get_bilineal_value(dist_kern, idx_x+0.5, idx_y+0.5, imagen)

    return out_vals

def get_bicubic_val(mod_x, a = -1):
    
    if mod_x <= 1:
        val = (a+2)*(mod_x**3) - (a+3)*(mod_x**2) + 1
    elif mod_x > 1 and mod_x < 2:
        val = (a)*(mod_x**3) - (5*a)*(mod_x**2) + 8*a*mod_x - 4*a
    else:
        val = 0
    return val

def calc_bicubic_point(imagen, distancias_x, distancias_y):
    
    aux_x_conv = np.zeros((4))
    
    

    # Interpolo en X
    for idx_y in range(4):
        for idx_x in range(4):
            mod_x = distancias_x[idx_x]
            aux_x_conv[idx_y] = aux_x_conv[idx_y] + get_bicubic_val(mod_x)*imagen[(idx_x*4)+idx_y]

        
    # Interpolo en Y
    out_val = 0
    for idx_y in range(4):
        mod_x = distancias_y[idx_y]
        out_val = out_val + get_bicubic_val(mod_x)*aux_x_conv[idx_y]
        
    return out_val
        
def func_downsample_bicubico(imagen, kernel = 1, a = -1):
    
    # El pixel lo considero central
    distancias = [1.5, 0.5, 0.5, 1.5]
    
    return calc_bicubic_point(imagen, distancias, distancias)



def func_upsample_bicubico(imagen, kernel = 1, a = -1):
    
    dist_kern = 2
    
    # Tengo varios puntos a samplear ahora, dados
    # por el tamaño del upsample
    out_vals = np.zeros((int(dist_kern),int(dist_kern)))
        
    # Las distancias van a estar moduladas segun
    # el pixel a samplear
    for idx_x in range(int(dist_kern)):
        for idx_y in range(int(dist_kern)):
            # Armo los vectores de distancias
            distancias_x = np.abs(np.array([1.5, 0.5, -0.5, -1.5])-0.25 + (0.5*idx_x))
            distancias_y = np.abs(np.array([1.5, 0.5, -0.5, -1.5])-0.25 + (0.5*idx_y))
            
            out_vals[idx_x,idx_y] = calc_bicubic_point(imagen, distancias_x, distancias_y)
                       
    return out_vals


def downsamplear(imagen, magnitud, modo = 'constante'):
    
    
    
    if modo == 'constante':
        func_use = func_downsample_hold
        kernel_use = np.ones((magnitud,magnitud))
        idx_row_base, idx_col_base = gen_downsample_indexes(magnitud)
        
    elif modo == 'bilineal':
        if magnitud % 2:
            print('Solo pasos multiplos de 2 son soportados.')
            return np.nan    
        func_use = func_downsample_bilineal
        kernel_use = np.ones((magnitud,magnitud))*magnitud
        idx_row_base, idx_col_base = gen_corners_indexes(magnitud)
        
    elif modo == 'bicubico':
        if magnitud != 2:
            print('Solo paso de 2 es soportado.')
            return np.nan    
        func_use = func_downsample_bicubico
        # En este caso el kernel debe ser de 4x4
        kernel_use = np.ones((4,4))
        idx_row_base, idx_col_base = C2D.gen_kern_relative_idx(kernel_use,CHECK_SHAPE = False)
       
    else:
        print ('Modo no soportado')
        return 
                          
    return C2D.conv_2D(imagen, 
                       kernel_use, 
                       paso_base = magnitud,
                       kernel_function = func_use,
                       idx_row_base = idx_row_base,
                       idx_col_base = idx_col_base)
    
    

def upsamplear(imagen, magnitud, modo = 'constante'):
    
    
    
    if modo == 'constante':
        func_use = func_upsample_hold
        kernel_use = np.ones((2,2))
        idx_row_base, idx_col_base = gen_upsample_indexes(magnitud)
        
    elif modo == 'bilineal':
        if magnitud % 2:
            print('Solo pasos multiplos de 2 son soportados.')
            return np.nan    
        func_use = func_upsample_bilineal
        kernel_use = np.ones((magnitud,magnitud))*magnitud
        idx_row_base, idx_col_base = gen_upsample_indexes(magnitud)
        
    elif modo == 'bicubico':
        if magnitud != 2:
            print('Solo paso de 2 es soportado.')
            return np.nan    
        func_use = func_upsample_bicubico
        # En este caso el kernel debe ser de 4x4
        kernel_use = np.ones((4,4))
        idx_row_base, idx_col_base = C2D.gen_kern_relative_idx(kernel_use,CHECK_SHAPE = False)
        
    else:
        print ('Modo no soportado')
        return 
                          
    return C2D.conv_2D(imagen, 
                       kernel_use, 
                       rango_out = magnitud,
                       kernel_function = func_use,
                       idx_row_base = idx_row_base,
                       idx_col_base = idx_col_base)

   
    
    
    
    

def func_scan_line(valor, escala, error):
    
    return escala[np.argmin((escala-valor+error)**2)]
    

def func_dithring(valor, escala, xxx):
    
    # Saco las distancias al color actual
    dist2color = (escala-valor)**2
    # Busco los dos colores mas cercanos
    arg_prim_min = np.argmin(dist2color)
    primer_min = escala[arg_prim_min]
    dist2color[arg_prim_min] = 99
    arg_seg_min = np.argmin(dist2color)
    segundo_min = escala[arg_seg_min]
    
    if primer_min < segundo_min:
        limites = [primer_min, segundo_min]
    else:
        limites = [segundo_min, primer_min]
    
    # Saco la distancia al menor
    dist2min = valor-limites[0]
    # Saco la proporcion del semirango
    prop = dist2min/(limites[1]-limites[0])
    
    # sorteo un valor
    if (np.random.rand() > (1-prop)):
        return limites[1]
    else:
        return limites[0]
    
    
    

def Cuantizar_gris(imagen, escalas, modo = 'uniforme'):
    
    imagen_use = np.zeros(imagen.shape)
    
    # Armo una escala de colores a utilizar
    valores_use = np.linspace(0,1,escalas) 
    
    # Elijo la funcion a aplicar
    if modo == 'uniforme':
        func_q = lambda valor, escala, x: escala[np.argmin((escala-valor)**2)]
    elif modo == 'dithering':
        func_q = func_dithring
    elif modo == 'scan-line':
        func_q = func_scan_line
    else:
        print('Metodo no encontrado')
        return np.nan
    
    # Recorro todos los pixeles
    (num_x, num_y) = imagen_use.shape
    for idx_x in range(num_x):
        error_act = 0
        for idx_y in range(num_y):
            imagen_use[idx_x, idx_y] = func_q(imagen[idx_x, idx_y], valores_use, error_act)
            error_act = error_act + (imagen_use[idx_x, idx_y]-imagen[idx_x, idx_y])
            
    
    return imagen_use