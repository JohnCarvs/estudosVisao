#------------------------------------------------------------------
# Autor: João Vitor de Carvalho Silva
#
# Descrição: O código usa um modelo de visão, o yolov8 para
#   fazer detecção de objetos em uma série de imagens. Tais
#   imagens estão em um diretório 'folder', o qual é integralmente
#   carregado para o detector. Instâncias da detecção são
#   visualizadas com cv2, e salvas com a função model() do YOLO
#   em um outro diretório.
# #------------------------------------------------------------------

import os
from ultralytics import YOLO
import cv2

#Modelo utilizado é carregado. O modelo 'yolov8n.pt' também funciona, mas possui mais falsos negativos
model = YOLO('../yoloWeights/yolov8n.pt')
model.info()

print(model.names)

#Para as imagens na pasta
folder = 'C:/Users/USUARIO/Documents/PyCharmProjectsC/objectDetection101/yoloProjects/yoloProject/RunningYolo/Images/Raw'
for nome_arquivo in os.listdir(folder):
    #Abre a imagem com yolo e gera os resultados, e abre a imagem com cv2
    results = model("Images/Raw/"+nome_arquivo, save=True)
    frame = cv2.imread(f'Images/Raw/'+nome_arquivo)

    #Iterar por detecções encontradas para desenhar as detecções
    for result in results:
        boxes = result.boxes
        for box in boxes:
            #Get coordenadas da bbox, threshold e a classe
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pt1 = tuple(map(int, (x1,y1)))
            pt2 = tuple(map(int, (x2,y2)))

            score = box.conf[0].item()
            class_id = int(box.cls[0].item())
            print(f'Bounding box: ({x1}, {y1}, {x2}, {y2}), Confiança: {score}, Classe: {model.names[class_id]}')

            #Desenhar o retângulo e o texto
            cv2.rectangle(frame, pt1 , pt2, (0, 255, 0), 2)
            cv2.putText(frame, (model.names[class_id] + " " + str(round(score,2))), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 2550, 0), 2)

    cv2.imshow("Leitura de texto", frame)
    cv2.waitKey(1000)
