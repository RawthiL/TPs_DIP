#!/usr/bin/python
# -*- coding: iso-8859-15 -*-



import numpy as np
from scipy import ndimage


import espacios_color as espc



def suma_y_rescalo(imag_1, imag_2, coef_1=1, coef_2=1, modo='RGB', saturacion='escalar' ):
    
    # Paso a float si no esta
    if imag_1.max() > 1.0:
        imag_1 = imag_1.astype(np.float32)/255.0
    if imag_2.max() > 1.0:
        imag_2 = imag_2.astype(np.float32)/255.0
    
    # Las hago mas o menos importante segun los parametros que pasé
    imag_1 = imag_1.astype(np.float32)*coef_1/(coef_1+coef_2)
    imag_2 = imag_2.astype(np.float32)*coef_2/(coef_1+coef_2)
    
    idx_1 = imag_1.shape
    idx_2 = imag_2.shape
    
    
    # La imagen que guardo es la que tiene mas pixeles
    image_aux = np.copy(imag_1) if (idx_2[0]*idx_2[1] < idx_1[0]*idx_1[1]) else np.copy(imag_2)
    
    # Busco el centro de las imagenes
    # Imagen final
    center_px = [0,0]
    center_px[0] = int(image_aux.shape[0]/2.0)
    center_px[1] = int(image_aux.shape[1]/2.0)
    # Imagen 1
    center_px_1 = [0,0]
    center_px_1[0] = int(idx_1[0]/2.0)
    center_px_1[1] = int(idx_1[1]/2.0)
    # Imagen 2
    center_px_2 = [0,0]
    center_px_2[0] = int(idx_2[0]/2.0)
    center_px_2[1] = int(idx_2[1]/2.0)


    # Me voy a detener en los margenes, en el primer margen que encuentre
    loop_lims = [0,0,0]
    for idx_aux in range(3):
        if idx_1[idx_aux] < idx_2[idx_aux]:
            loop_lims[idx_aux] = idx_1[idx_aux]
        else:
            loop_lims[idx_aux] = idx_2[idx_aux]

    if modo=='RGB':
        # Sumo todos los canales...
        for idx_aux_x in range(loop_lims[0]):
            for idx_aux_y in range(loop_lims[1]):
                for idx_aux_z in range(loop_lims[2]):
                    image_aux[center_px[0]-int(loop_lims[0]/2)+idx_aux_x, 
                              center_px[1]-int(loop_lims[1]/2)+idx_aux_y, 
                              idx_aux_z]                           = imag_1[center_px_1[0]-int(loop_lims[0]/2)+idx_aux_x, 
                                                                            center_px_1[1]-int(loop_lims[1]/2)+idx_aux_y, 
                                                                            idx_aux_z]                             + imag_2[center_px_2[0]-int(loop_lims[0]/2)+idx_aux_x, 
                                                                                                                            center_px_2[1]-int(loop_lims[1]/2)+idx_aux_y, 
                                                                                                                            idx_aux_z]

    elif modo == 'YIQ':
        # Sumo vectorialmente HACELO!!!! Tenes que sumar los vectores
        # Que parten de cero y van hasta el plano de cromicidad
        # despues se suman vectorialmente
        for idx_aux_x in range(loop_lims[0]):
            for idx_aux_y in range(loop_lims[1]):
                idx_use_x_0 = center_px[0]-int(loop_lims[0]/2)+idx_aux_x
                idx_use_y_0 = center_px[1]-int(loop_lims[1]/2)+idx_aux_y
                idx_use_x_1 = center_px_1[0]-int(loop_lims[0]/2)+idx_aux_x
                idx_use_y_1 = center_px_1[1]-int(loop_lims[1]/2)+idx_aux_y
                idx_use_x_2 = center_px_2[0]-int(loop_lims[0]/2)+idx_aux_x
                idx_use_y_2 = center_px_2[1]-int(loop_lims[1]/2)+idx_aux_y
                
                Y_1 = imag_1[idx_use_x_1,idx_use_y_1,0]
                Y_2 = imag_2[idx_use_x_2,idx_use_y_2,0]
                                
                image_aux[idx_use_x_0,idx_use_y_0,0] = Y_1+Y_2
                
                # I y Q se escalan con Y, ya que mantienen su valor aunque cambie la luminancia.
                # Sumarlos no se puede, hay que escalarlos segun la luminancia antes...
                image_aux[idx_use_x_0,idx_use_y_0,1] = (Y_1 * imag_1[idx_use_x_1,idx_use_y_1,1] + Y_2 * imag_2[idx_use_x_2,idx_use_y_2,1] )/ ( Y_1+Y_2 )
                image_aux[idx_use_x_0,idx_use_y_0,2] = (Y_1 * imag_1[idx_use_x_1,idx_use_y_1,2] + Y_2 * imag_2[idx_use_x_2,idx_use_y_2,2] )/ ( Y_1+Y_2 )


    
    # Lo traigo devuelta a sus limites

    if saturacion=='escalar':
        image_aux = image_aux/(2)
    elif saturacion=='full_range':
        image_aux = image_aux/image_aux.max()
    elif saturacion=='crop':
        if modo == 'RGB':
            image_aux = espc.check_RGB(image_aux, modo = 'float')
        elif modo == 'YIQ':
            image_aux = espc.check_YIQ(image_aux)
        else:
            print('Espacio de color no valido.')
    else:
        print('Modo de saturacion no valido.')   

    
    return image_aux




def resta_y_rescalo(imag_1, imag_2, coef_1=1, coef_2=1, modo='RGB', saturacion='escalar' ):
    
    # Paso a float si no esta
    if imag_1.max() > 1.0:
        imag_1 = imag_1.astype(np.float32)/255.0
    if imag_2.max() > 1.0:
        imag_2 = imag_2.astype(np.float32)/255.0
    
    # Las hago mas o menos importante segun los parametros que pasé
    imag_1 = imag_1.astype(np.float32)*coef_1/(coef_1+coef_2)
    imag_2 = imag_2.astype(np.float32)*coef_2/(coef_1+coef_2)
    
    idx_1 = imag_1.shape
    idx_2 = imag_2.shape
   
    # La imagen que guardo es la que tiene mas pixeles
    image_aux = np.copy(imag_1) if (idx_2[0]*idx_2[1] < idx_1[0]*idx_1[1]) else np.copy(imag_2)
    
    # Busco el centro de las imagenes
    # Imagen final
    center_px = [0,0]
    center_px[0] = int(image_aux.shape[0]/2.0)
    center_px[1] = int(image_aux.shape[1]/2.0)
    # Imagen 1
    center_px_1 = [0,0]
    center_px_1[0] = int(idx_1[0]/2.0)
    center_px_1[1] = int(idx_1[1]/2.0)
    # Imagen 2
    center_px_2 = [0,0]
    center_px_2[0] = int(idx_2[0]/2.0)
    center_px_2[1] = int(idx_2[1]/2.0)


    # Me voy a detener en los margenes, en el primer margen que encuentre
    channels = len(image_aux.shape)
    loop_lims = np.zeros((channels), dtype=np.int)
    for idx_aux in range(channels):
        if idx_1[idx_aux] < idx_2[idx_aux]:
            loop_lims[idx_aux] = idx_1[idx_aux]
        else:
            loop_lims[idx_aux] = idx_2[idx_aux]


    if modo=='RGB':
        # Resto todos los canales...
        for idx_aux_x in range(loop_lims[0]):
            for idx_aux_y in range(loop_lims[1]):
                for idx_aux_z in range(loop_lims[2]):
                    image_aux[center_px[0]-int(loop_lims[0]/2)+idx_aux_x, 
                              center_px[1]-int(loop_lims[1]/2)+idx_aux_y, 
                              idx_aux_z]                           = imag_1[center_px_1[0]-int(loop_lims[0]/2)+idx_aux_x, 
                                                                            center_px_1[1]-int(loop_lims[1]/2)+idx_aux_y, 
                                                                            idx_aux_z]                             - imag_2[center_px_2[0]-int(loop_lims[0]/2)+idx_aux_x, 
                                                                                                                            center_px_2[1]-int(loop_lims[1]/2)+idx_aux_y, 
                                                                                                                            idx_aux_z]

    elif modo == 'YIQ':
        for idx_aux_x in range(loop_lims[0]):
                for idx_aux_y in range(loop_lims[1]):
                    idx_use_x_0 = center_px[0]-int(loop_lims[0]/2)+idx_aux_x
                    idx_use_y_0 = center_px[1]-int(loop_lims[1]/2)+idx_aux_y
                    idx_use_x_1 = center_px_1[0]-int(loop_lims[0]/2)+idx_aux_x
                    idx_use_y_1 = center_px_1[1]-int(loop_lims[1]/2)+idx_aux_y
                    idx_use_x_2 = center_px_2[0]-int(loop_lims[0]/2)+idx_aux_x
                    idx_use_y_2 = center_px_2[1]-int(loop_lims[1]/2)+idx_aux_y

                    Y_1 = imag_1[idx_use_x_1,idx_use_y_1,0]
                    Y_2 = imag_2[idx_use_x_2,idx_use_y_2,0]

                    image_aux[idx_use_x_0,idx_use_y_0,0] = Y_1-Y_2

                    # I y Q se escalan con Y, ya que mantienen su valor aunque cambie la luminancia.
                    # Sumarlos no se puede, hay que escalarlos segun la luminancia antes...
                    image_aux[idx_use_x_0,idx_use_y_0,1] = (Y_1 * imag_1[idx_use_x_1,idx_use_y_1,1] - Y_2 * imag_2[idx_use_x_2,idx_use_y_2,1] )/ ( Y_1+Y_2 )
                    image_aux[idx_use_x_0,idx_use_y_0,2] = (Y_1 * imag_1[idx_use_x_1,idx_use_y_1,2] - Y_2 * imag_2[idx_use_x_2,idx_use_y_2,2] )/ ( Y_1+Y_2 )

    elif modo == 'LUMIN': 
        for idx_aux_x in range(loop_lims[0]):
                for idx_aux_y in range(loop_lims[1]):
                    idx_use_x_0 = center_px[0]-int(loop_lims[0]/2)+idx_aux_x
                    idx_use_y_0 = center_px[1]-int(loop_lims[1]/2)+idx_aux_y
                    idx_use_x_1 = center_px_1[0]-int(loop_lims[0]/2)+idx_aux_x
                    idx_use_y_1 = center_px_1[1]-int(loop_lims[1]/2)+idx_aux_y
                    idx_use_x_2 = center_px_2[0]-int(loop_lims[0]/2)+idx_aux_x
                    idx_use_y_2 = center_px_2[1]-int(loop_lims[1]/2)+idx_aux_y

                    Y_1 = imag_1[idx_use_x_1,idx_use_y_1]
                    Y_2 = imag_2[idx_use_x_2,idx_use_y_2]

                    image_aux[idx_use_x_0,idx_use_y_0] = Y_1-Y_2

        
    else:
        print('Espacio de color no valido.')

    
    # Lo traigo devuelta a sus limites
    
    if saturacion=='full_range':
        if modo == 'RGB' or modo == 'LUMIN': 
            image_aux = (image_aux-image_aux.min())
            image_aux = image_aux/image_aux.max()
        elif modo == 'YIQ':
            image_aux[:,:,0] = (image_aux[:,:,0]-image_aux[:,:,0].min())
            image_aux[:,:,0] = image_aux[:,:,0]/image_aux[:,:,0].max()
    elif saturacion=='crop':
        if modo == 'RGB':
            image_aux = espc.check_RGB(image_aux, modo = 'float')
        elif modo == 'YIQ':
            image_aux = espc.check_YIQ(image_aux)
        elif modo == 'LUMIN': 
            image_aux = espc.check_LUMINANCIA(image_aux, modo = 'float')
       
        
    
    
    return image_aux