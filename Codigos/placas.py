# -*- coding: utf-8 -*-
"""--------------------------------------------------------------------
--------------Detección de Placas Amarillas de Carros------------------
-----------------------------------------------------------------------
----------------------CHRISTIAN GALLEGO CHAVERRA-----------------------
------------------chrsitian.gallego@udea.edu.co------------------------
----------------------------CC: 1017214040-----------------------------
-----------------------------------------------------------------------
----------------------CAMILO ENRIQUE FARELO PANESSO--------------------
----------------------camilo.farelo@udea.edu.co------------------------
----------------------------CC: 1093793316 ----------------------------
-----------------------------------------------------------------------
----Curso Básico de Procesamiento de Imágenes y Visión Artificial------
--------------------------Octubre de 2020------------------------------
-----------------------------------------------------------------------
"""

from __future__ import division

#Valores empiricos de ubicacion y area de los caracteres en las placas
#necesarios para la busqueda de los caracteres en la placa
#Se escogieron debido a que todas las placas tienen la misma forma y ubicacion de los caracteres, generalmente
#areas maximas y minimas que ocupan los caracteres
min_area = 2.1
max_area = 5.9
#anchos
min_width = 12.0
max_width = 14.0
#altos
min_height = 60
max_height = 55

#posicion inicial en y
min_y0 = 20.5

#Posiciones de los caracteres en la placa
x0_word_1 = 81.7
x0_word_2 = 67.7
x0_word_3 = 53.4
x0_num_1 = 33.5
x0_num_2 = 19.5
x0_num_3 = 5

ofset_x0 = [x0_word_1, x0_word_2, x0_word_3, x0_num_1, x0_num_2, x0_num_3]

#Librerias
import numpy as np
import cv2
import glob, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Carpeta con las imagenes de los carros
folder = 'C:/Users/Christian Gallego/Desktop/PDIfinal/carros'
folderMuestras = 'C:/Users/Christian Gallego/Desktop/PDIfinal/muestras'

os.chdir(folder)
files = glob.glob('*')

#Usando una proporcion geometrica, esta funcion detecta si el contorno es un rectangulo
def detectar(c):
    #Variable para comprobar la forma
    ar = 0.0 
    #Variable donde se guarda la forma encontrada, empieza no definida
    shape = "unidentified"
    #Capturamos el perimetro de la imagen
    perimetro = cv2.arcLength(c, True)
    #Suaviza y aproxima al cuadrilatero
    aproximado = cv2.approxPolyDP(c, 0.04 * perimetro, True)
    
    #Analizamos el resultado del aproximado, si es 4 es un cuadrado o rectangulo
    if len(aproximado) == 4:
        #Capturamos las medidas del rectangulo
        (x,y,w,h) = cv2.boundingRect(aproximado)
        ar = float(w)/float(h)
    #endif
    #Comprobamos si es un cuadrado o un rectangulo
    if ar >= 0.96 and ar <= 1:
        shape = "square" 
    else: 
        shape = "rectangle"
    #endif
    #retornamos la forma
    return shape

#Esta funcion retorna un box con la placa 
#Recibe como parametro una imagen binarizada
#La imagen original escalada
#Y el contorno de la figura que se usa para calcular el area
def encuentra_placa(img_bin, img_reszd, contornos):
    mayor = 0;
    #se recorren los contornos para encontrar la mayor area
    for h,cnt in enumerate(contornos):
        mask = np.zeros(img_bin.shape,np.uint8)#mascara de negros
        cv2.drawContours(mask, [cnt],0,255,-1)
        area = cv2.contourArea(cnt)
        if area > mayor:
            mayor = area
        #endif
    #endfor
    
    #Se recorren de nuevo todos los contornos encontrados en cada imagen
    #buscando la mayor area y que cumpla proporcion geometrica de los rectangulos
    for h,cnt in enumerate(contornos):
        mask = np.zeros(img_bin.shape,np.uint8)#mascara de negros
        cv2.drawContours(mask, [cnt],0,255,-1)
        area = cv2.contourArea(cnt)
        #Se limita el area a unos pixeles especificos de los cuales se sabe que puede ser el area de la placa
        #luego se compara si es el area mayo y si tiene forma rectangular
        if area > 7000:
            if area < 150000:
                form = detectar(cnt)
                #Si cumple que no hay areas mas pequeñas y que es un rectangulo, suponemos que encontramos la placa
                if area == mayor and form == 'rectangle' :
                    x,y,w,b = cv2.boundingRect(cnt)
                    box = img_reszd[y:y+b,x:x+w]
                    return box
                
#Lectura de la imagen
for file in files:    
    img = cv2.imread(folder + "\\" + file) 
    
    #cv2.imshow("original", img)
     
    #Reajusta el tamaño de la imagen  
    img_reszd = img[800:2050, 1100:2200] 
    
    #cv2.imshow("resziced", img_reszd)
    
    #Cambia la imagen al espacio LAB
    img_lab = cv2.cvtColor(img_reszd, cv2.COLOR_BGR2LAB)
    
    #cv2.imshow("lab", img_lab)
    
    #Toma la componente roja de la imagen
    img_red = img_lab[:,:,2]
    
    #cv2.imshow("red", img_red)
    
    #Se binariza la imagen
    ret,img_bin = cv2.threshold(img_red.copy(), 140,255,cv2.THRESH_BINARY)
    
    #Capturamos los contornos de la misma
    contornos, pos = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Aplicamos la funcion para encontrar la placa
    box = encuentra_placa(img_bin, img_reszd, contornos)
    
    #Guardamos la imagen de solo la placa 
    img = box
    
    # cv2.imshow('placa', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ######################################################################################
    #cambio de espacio de color lab
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #Tomamos la componente roja de la imgen de la placa
    img_red = img_lab[:,:,2]
    #Extraemos la imagen binarizada
    ret, img_bin = cv2.threshold(img_red,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    height, width, channels = img.shape
    totalArea = height*width
    
    #Contorno de la imagen binarizada
    contornos, hierarchy = cv2.findContours(img_bin,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    #Vector asociativo para las caracteristicas respecto a cada placa
    features = {'promX': 0, 'promW': 0, 'promY': 0, 'promH': 0}
    #Se realiza una mascara con los contornos y se extraen las mas apropiadas que sean caracteres
    for h, cnt in enumerate(contornos):
        #mascara de negros
        mask = np.zeros(img_bin.shape,np.uint8) 
        mask = cv2.resize(mask, (400,400))
        #contornos de la mascara
        cv2.drawContours(mask,[cnt],0,255,-1)
        #area de las mascaras
        area = cv2.contourArea(cnt)
      
        #Se calcula que las areas de los contornos sean las apropiadas para almacenar caracteres
        porcentajeArea = round((area/totalArea)*100, 2)
        #filtrado en el area delimitada por los valores empiricos
        if (porcentajeArea > min_area and porcentajeArea < max_area):
            x,y,w,b = cv2.boundingRect(cnt)
            if (w/width*100) < (max_width+7):
              porcentajeWidth = (w/width)*100
              box = img[y:y+b, x:x+w]
              #extraemos promedios de las dimensiones 
              features['promX'] += x
              features['promY'] += y
              features['promW'] += w
              features['promH'] += b
              chars.append({'img':box, 'x':(x/width)*100, 'box': {'x':x,'y':y,'w':w,'h':h}})
      
        #Se realiza un calculo de caracteristiacas especificas o generales para realizar la extraccion de caracteres
        imgLen = len(chars)
        if imgLen > 0:
            features['promX'] = features['promX']/imgLen
            features['promY'] = features['promY']/imgLen
            features['promW'] = features['promW']/imgLen
            features['promH'] = features['promH']/imgLen
        #Si no extrae bien las dimensiones, usamos los valores empiricos
        else:
            features['promY'] = (min_y0-5)/100*height
            features['promW'] = (min_width + max_width)/2/100*width
            features['promH'] = (min_height + max_height)/2/100*height
        
        #endif
        
        #Se ordenan loscaracteres extraidos por medio de contornos
        charPosicion = [None,None,None,None,None,None]
        fixOfset = min_width/2
        #Ubicamos los caracteres en orden de acuerdo a la posicion en la que se encuentre en un array
        for char in chars:
          if (x0_word_1 - fixOfset) <= char['x'] <= (x0_word_1 + fixOfset):
              charPosicion[0] = [char['img'], char['box']]
          elif (x0_word_2 - fixOfset) <= char['x'] <= (x0_word_2 + fixOfset):
              charPosicion[1] = [char['img'], char['box']]
          elif (x0_word_3 - fixOfset) <= char['x'] <= (x0_word_3 + fixOfset):
              charPosicion[2] = [char['img'], char['box']]
          elif (x0_num_1 - fixOfset) <= char['x'] <= (x0_num_1 + fixOfset):
              charPosicion[3] = [char['img'], char['box']]
          elif (x0_num_2 - fixOfset) <= char['x'] <= (x0_num_2 + fixOfset):
              charPosicion[4] = [char['img'], char['box']]
          elif (x0_num_3 - fixOfset) <= char['x'] <= (x0_num_3 + fixOfset):
              charPosicion[5] = [char['img'], char['box']]
          #endif
        #endfor
        #Se extraen los caracteres por medio de caracteristicas definidas en la propia placa o las sacadas empiricamente
        #Esto si la funcion contornos falla y no saca bien los caracteres
        for i in range(0, 6):
          if charPosicion [i] == None:
              x = int(ofset_x0[i]/100*width)
              y = int(features['promY'])
              w = int(ofset_x0[i]/100*width+features['promW'])
              h = int(features['promY']+features['promW'])
              charPosicion[i]=[img_lab[y:h, x:w], {'x':x,'y':y,'w':w-x,'h':h-y}]
          #endif
        #endfor
        
        placa = ''
        fileMatch = ''
        
        os.chdir(folderMuestras)
        compFiles = glob.glob('*')
        
        for char in charPosicion:
            fileMatch = ''
            inLower = None
            inBigger = None
            lower = 9999999999999
            bigger = -9999999999999
            for compFile in compFiles:
                imComp = cv2.imread(folderMuestras + "\\" + compFile)
                y = char[1]['y']
                h = y + char[1]['h']
                x = char[1]['x']
                w = x + char[1]['w']
                #clone = img_bin[y:h, x:w].copy()
                compH, compW = imComp.shape[:2]
                #clone = cv2.resize(clone, (compW, compH), interpolation=cv2.INTER_CUBIC)
                value = 0
                
                if value < lower:
                    lower = value
                
                #endif
                if value > bigger:
                    bigger = value
                    fileMatch = compFile
                #endif   
            placa += fileMatch.split('.')[0]
        
        #Se imprimen las imagenes
        cv2.imshow('digito6',charPosicion[5][0])
        cv2.imshow('digito5',charPosicion[4][0])
        cv2.imshow('digito4',charPosicion[3][0])
        cv2.imshow('digito3',charPosicion[2][0])
        cv2.imshow('digito2',charPosicion[1][0])
        cv2.imshow('digito1',charPosicion[0][0])                          
    #endfor
#endfor
cv2.waitKey(0)
cv2.destroyAllWindows()