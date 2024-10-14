# Autor: João Vitor de Carvalho Silva
#
# Descrição: O programa conta a quantidade de carros que passa
# em uma avenida. O modelo de detecção de objetos YOLO v8 é usado,
# bem como um tracker. Os IDs de cada instância de veículo é salva
# em um set().
# #------------------------------------------------------------------

import random
from random import randint
import math
import numpy as np
from ultralytics import YOLO
import cv2 as cv2
import cvzone
from sort import *


#Função para gerar cores aleatórias de acordo com uma entrada de texto. Usada para display das cores
def randomDrawColor(cls):
    random.seed(cls)
    colorArray = np.array([randint(0,255) for i in range(3)])
    colorArray = list(int(i) for i in ((colorArray/colorArray.max())**1.2)*255)
    return (colorArray)

#Função para gerar cores para textos, com base na cor do background. Usada para melhorar legibilidade
def textColor(backgroundColor):
    value = sum(backgroundColor)/3
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

#Configurar input
visualInput = cv2.VideoCapture("../videos/cars.mp4") #Videos
visualInput.set(3, 1280)
visualInput.set(4, 720)
mask = cv2.imread("mask.jpg")

#Configurar o tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#Definir a linha de contagem
limits = [[140,407],[673,497]]

#Criar lista de IDs
carsIds = set()

while True:
    success, frame = visualInput.read()
    imRegion = cv2.bitwise_and(frame, mask)

    results = model(imRegion, stream=True)

    detections = np.empty((0,5))

    for result in results:
        boxes = result.boxes
        for box in boxes:
            #Get coordenadas da bbox, threshold e a classe
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pt1 = tuple(map(int, (x1,y1)))
            pt2 = tuple(map(int, (x2,y2)))

            class_id = int(box.cls[0].item())
            className = model.names[class_id]

            score = box.conf[0].item()

            if ((className == "car" or className == "bus" or className == "truck" or className == "motorcycle") and (score > 0.4)) :
                colorB = randomDrawColor(class_id)
                #Desenha o ponto medial de

                #print(f'Bounding box: ({x1}, {y1}, {x2}, {y2}), Confiança: {score}, Classe: {model.names[class_id]}, Cor: {randomDrawColor(className)}')

                #Desenhar o retângulo e o texto
                cv2.rectangle(frame, pt1 , pt2, colorB, 1)
                cvzone.putTextRect(frame, f'{className} {round(score,2)}', tuple(map(int,(max(0,x1),max(40,y1)))), 0.9,1,textColor(colorB), colorB)

                currentArray = np.array([x1,y1,x2,y2,score])
                detections = np.vstack((detections,currentArray))

    trackerOutput = tracker.update(detections)
    cv2.line(frame, (limits[0][0],   int(np.mean([limits[0][1],limits[1][1]]))    ),(limits[1][0],int(np.mean([limits[0][1],limits[1][1]]))    ),(0,0,255),3)

    #Iterar para o tracker
    for trackerInstance in trackerOutput:
        x1,y1,x2,y2,Id = trackerInstance
        medialPoint = tuple(map(int,( np.mean([x1,x2]) , np.mean([y1,y2])) ))

        if isPointInInterval(medialPoint, limits):
            carsIds.add(Id)

        #cv2.putText(frame,f"{int(Id)}", tuple(map(int,(x2,y1))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cvzone.putTextRect(frame, f"id: {int(Id)}", tuple(map(int,(x2,y2))), 1, 0, (255,255,255), (0,0,0))


    #Printar a contagem total
    cvzone.putTextRect(frame, f"count: {len(carsIds)}", ((100,50)), 2, 0, (255,255,255), (0,0,0))

    cv2.imshow("Output", frame)
    cv2.waitKey(1)
