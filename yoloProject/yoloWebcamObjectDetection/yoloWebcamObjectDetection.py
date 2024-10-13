import random
from random import randint
import math
import numpy as np
from numpy.random import randn
from ultralytics import YOLO
import cv2 as cv2
import cvzone


#Função para gerar cores aleatórias de acordo com uma entrada de texto. Usada para display das cores
def randomDrawColor(cls):
    #seed = sum( ord(char) + 1  for char in (list(className)))
    random.seed(cls)
    colorArray = np.array([randint(0,255) for i in range(3)])
    colorArray = list(int(i) for i in ((colorArray/colorArray.max())**1.2)*255)
    return (colorArray)

#Função para gerar cores para textos, com base na cor do background. Usada para melhorar legibilidade
def textColor(backgroundColor):
    value = sum(backgroundColor)/3
    print (value)
    return (tuple(   ((math.floor(value/140))*-255)+255 for _ in range(3)))




#Modelo utilizado é carregado. O modelo 'yolov8n.pt' também funciona, mas possui mais falsos negativos
model = YOLO('../yoloWeights/yolov8n.pt')
model.info()

print( model.names)

#Configurar camera input
#camera = cv2.VideoCapture(0) #Webcam
camera = cv2.VideoCapture("../videos/bikes.mp4") #Videos
camera.set(3,1280)
camera.set(4, 720)

while True:
    success, frame = camera.read()
    results = model(frame, stream=True)


    for result in results:
        boxes = result.boxes
        for box in boxes:
            #Get coordenadas da bbox, threshold e a classe
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pt1 = tuple(map(int, (x1,y1)))
            pt2 = tuple(map(int, (x2,y2)))

            score = box.conf[0].item()
            class_id = int(box.cls[0].item())
            className = model.names[class_id]

            colorB = randomDrawColor(class_id)

            print(f'Bounding box: ({x1}, {y1}, {x2}, {y2}), Confiança: {score}, Classe: {model.names[class_id]}, Cor: {randomDrawColor(className)}')

            #Desenhar o retângulo e o texto
            cv2.rectangle(frame, pt1 , pt2, colorB, 2)
            cvzone.putTextRect(frame, f'{className} {round(score,2)}', tuple(map(int,(max(0,x1),max(40,y1)))), 1.3,1,textColor(colorB), colorB)


    cv2.imshow("Camera", frame)
    cv2.waitKey(1)
