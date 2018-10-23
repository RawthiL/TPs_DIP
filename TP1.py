from scipy import ndimage
import skimage as ski
import numpy as np
import os, sys
import matplotlib.pyplot as plt

#TEST_FOLDER = './Imagenes'
#TEST_IMAGE = 'spacex-rocket-pictures.jpg'
#ALPHA_VAL = 0.5
#BETA_VAL = 1.2
#nom_arch = os.path.join(TEST_FOLDER,TEST_IMAGE)

nom_arch = sys.argv[1]
ALPHA_VAL = float(sys.argv[2])
BETA_VAL = float(sys.argv[3])
if len(sys.argv) >= 5:
    PLOT_ALL = int(sys.argv[4])
else:
    PLOT_ALL = False




imag_ini = ndimage.imread(nom_arch, flatten=False, mode='RGB')




RGB2YIQ = np.array([[0.299,      0.587,        0.114],  [0.59590059, -0.27455667, -0.32134392],  [0.21153661, -0.52273617, 0.31119955]])

def check_RGB(imagen_RGB):
    
    image_shape = imagen_RGB.shape
    num_px = image_shape[0]*image_shape[1]

    lista_fix = np.argwhere((imagen_RGB[:,:,0] > 255 ) | 
                            (imagen_RGB[:,:,1] > 255 ) | 
                            (imagen_RGB[:,:,2] > 255 ) | 
                            (imagen_RGB[:,:,0] < 0.0 ) | 
                            (imagen_RGB[:,:,1] < 0.0 ) | 
                            (imagen_RGB[:,:,2] < 0.0 ) )
    for px in lista_fix:
        idx_x = px[0]
        idx_y = px[1]
        for i in range(3):
            if imagen_RGB[idx_x,idx_y,i] < 0:
                imagen_RGB[idx_x,idx_y,i] = 0
            elif imagen_RGB[idx_x,idx_y,i] > 255:
                imagen_RGB[idx_x,idx_y,i] = 255
        # obtengo la recta al centro de coordenadas
        #P = imagen_RGB[idx_x,idx_y,:] - [0,0,0]
        # Resuelvo para el punto que primero sature
        #a = np.array([0,0,0])
        #for i in range(3):
        #    if imagen_RGB[idx_x,idx_y,i] < 0:
        #        a[i] =    0.0 / P[i]
        #    else:
        #        a[i] =  255.0 / P[i]
        #a[np.argwhere(np.isnan(a))] = 9e9
        #a[np.argwhere(np.isinf(a))] = 9e9
        #idx_a_use = np.argmin(np.abs(a))
        # Obtengo el valor saturado del pixel
        #imagen_RGB[idx_x,idx_y,:] = P*a[idx_a_use]
        #print(imagen_RGB[idx_x,idx_y,:])

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
        # obtengo la recta al centro de coordenadas
        P = imagen[idx_x,idx_y,:] - [0,0,0]
        # Resuelvo para el punto que primero sature
        a = [0,0,0,0,0]
        a[0] =           1.0 / P[0]
        a[1] =  RGB2YIQ[1,0] / P[1]
        a[2] = -RGB2YIQ[1,0] / P[1]
        a[3] =  RGB2YIQ[2,1] / P[2]
        a[4] = -RGB2YIQ[2,1] / P[2]
        idx_a_use = np.argmin(np.abs(a))
        # Obtengo el valor saturado del pixel
        imagen[idx_x,idx_y,:] = P*a[idx_a_use]

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



def rgb2yiq(imagen):
    # Las componentes RGB en la matriz ndimensional que representa 
    # la imagen parecen estar puestas como dimension fila cuando
    # numpy hace la multiplicación, así que tenemos que 
    # transponer la matriz de conversión
    imagen_use = np.double(imagen)/255.0
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


imag_yiq = rgb2yiq(imag_ini)


#plt.figure(dpi=300)
#ax = plt.subplot(1,2,1)
#ax.imshow(imag_ini)
#ax = plt.subplot(1,2,2)
#ax.imshow(imag_yiq[:,:,0],  cmap='gray')
#plt.axis('off')
#plt.show()


if not PLOT_ALL:
    imag_yiq_alpha = aplicar_alpha(imag_yiq, ALPHA_VAL)
    imag_yiq_beta = aplicar_beta(imag_yiq, BETA_VAL)


    imag_rgb_alpha =  yiq2rgb(imag_yiq_alpha)
    imag_rgb_beta =  yiq2rgb(imag_yiq_beta)


    plt.figure(dpi=300)
    ax = plt.subplot(2,2,1)
    ax.imshow(imag_ini)
    plt.axis('off')
    plt.title('Original')
    ax = plt.subplot(2,2,2)
    ax.imshow(imag_yiq[:,:,0],  cmap='gray')
    plt.axis('off')
    plt.title('Canal Y')
    ax = plt.subplot(2,2,3)
    ax.imshow(imag_rgb_alpha)
    plt.axis('off')
    plt.title('Alpha = %0.2f'%ALPHA_VAL)
    ax = plt.subplot(2,2,4)
    ax.imshow(imag_rgb_beta)
    plt.axis('off')
    plt.title('Beta = %0.2f'%BETA_VAL)

    plt.draw()
    plt.savefig('resultado.png')


    print(np.min(imag_rgb_beta))
    print(np.max(imag_rgb_beta))
    print(np.mean(imag_rgb_beta))
    print(np.std(imag_rgb_beta))


    plt.show()

else:
    alfas = [0.2, 0.5, 1.2, 1.7]
    betas = [0.2, 0.5, 1.2, 1.7]


    plt.figure(dpi=300)
    for idx in range(4):
        
        imag_yiq_alpha = aplicar_alpha(imag_yiq, alfas[idx])
        imag_rgb_alpha =  yiq2rgb(imag_yiq_alpha)

        ax = plt.subplot(2,2,idx+1)
        ax.imshow(imag_rgb_alpha)
        plt.axis('off')
        plt.title('Alpha: %0.2f'%alfas[idx])

    plt.draw()
    plt.savefig('alfas.png')

    plt.figure(dpi=300)
    for idx in range(4):
        
        imag_yiq_beta = aplicar_beta(imag_yiq, betas[idx])
        imag_rgb_beta =  yiq2rgb(imag_yiq_beta)

        ax = plt.subplot(2,2,idx+1)
        ax.imshow(imag_rgb_beta)
        plt.axis('off')
        plt.title('Beta: %0.2f'%betas[idx])

    plt.draw()
    plt.savefig('betas.png')


plt.show()