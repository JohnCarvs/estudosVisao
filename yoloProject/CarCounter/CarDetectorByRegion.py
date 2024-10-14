# Autor: João Vitor de Carvalho Silva
#
# Descrição: O código usa um modelo de visão, o yolov8 para
# fazer detecção de veúclos em uma avenida, dada uma região retangular
# do quadro de entrada.
# #------------------------------------------------------------------

import random
from random import randint
import math
import numpy as np
from numpy.random import randn
from torch.distributions.constraints import interval
from ultralytics import YOLO
import cv2 as cv2
import cvzone


#Função para gerar cores aleatórias de acordo com uma entrada de texto. Usada para display das cores
def randomDrawColor(cls):
    random.seed(cls)
    colorArray = np.array([randint(0,255) for i in range(3)])
    colorArray = list(int(i) for i in ((colorArray/colorArray.max())**1.2)*255)
    return (colorArray)

#Função para gerar cores para textos, com base na cor do background. Usada para melhorar legibilidade
def textColor(backgroundColor):
    value = sum(backgroundColor)/3
    print (value)
    return (tuple(   ((math.floor(value/140))*-255)+255 for _ in range(3)))

def isPointInInterval(pt,interval):
    print(pt)
    print(interval)
    if pt[0] in range (interval[0][0], interval[1][0]) and pt[1] in range (interval[0][1],interval[1][1]):
        return(True)
    else:
        return(False)

#Modelo utilizado é carregado. O modelo 'yolov8n.pt' também funciona, mas possui mais falsos negativos
model = YOLO('../yoloWeights/yolov8l.pt')
model.info()

print( model.names)

#Configurar camera input
#camera = cv2.VideoCapture(0) #Webcam
camera = cv2.VideoCapture("../videos/cars.mp4") #Videos
camera.set(3,1280)
camera.set(4, 720)

intervalOfCarCapture = [tuple((250,280)), tuple((700,700))]

while True:
    success, frame = camera.read()
    results = model(frame, stream=True)

    #Desenha o intervalo de captura de carros
    cv2.rectangle(frame, intervalOfCarCapture[0],intervalOfCarCapture[1], (255,255,0),1)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            #Get coordenadas da bbox, threshold e a classe
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pt1 = tuple(map(int, (x1,y1)))
            pt2 = tuple(map(int, (x2,y2)))
            medialPoint = tuple(map(int,( np.mean([x1,x2]) , np.mean([y1,y2])) ))

            class_id = int(box.cls[0].item())
            className = model.names[class_id]

            score = box.conf[0].item()
            print(medialPoint)
            print(f"teste de funcao: {isPointInInterval(medialPoint,intervalOfCarCapture)}")
            if ((className == "car" or className == "bus" or className == "truck" or className == "motorbike") and (score > 0.4) and (isPointInInterval(medialPoint,intervalOfCarCapture) == True)) :
                colorB = randomDrawColor(class_id)
                #Desenha o ponto medial de
                cv2.circle(frame,medialPoint,2,colorB, 3)

                print(f'Bounding box: ({x1}, {y1}, {x2}, {y2}), Confiança: {score}, Classe: {model.names[class_id]}, Cor: {randomDrawColor(className)}')

                #Desenhar o retângulo e o texto
                cv2.rectangle(frame, pt1 , pt2, colorB, 1)
                cvzone.putTextRect(frame, f'{className} {round(score,2)}', tuple(map(int,(max(0,x1),max(40,y1)))), 0.9,1,textColor(colorB), colorB)


    cv2.imshow("Camera", frame)
    cv2.waitKey(1)
