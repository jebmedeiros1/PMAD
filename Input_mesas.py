from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import pyodbc
from datetime import datetime
import gc  # Importando o módulo de coleta de lixo
 
# Configurações iniciais
server = '10.8.100.69'
database = 'GMA'
username = 'Spotfireread'
password = 'Consult@'
 
# Carregar o modelo e os rótulos apenas uma vez
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
 
# Capturar dimensões da tela
screen_width, screen_height = pyautogui.size()
 
# Desabilitar notação científica para clareza
np.set_printoptions(suppress=True)
 
# Definir bbox para capturar a tela inteira
initial_bbox = (0, 0, screen_width, screen_height)
 
def selecionar_rois(num_screens, initial_bbox):
    """Selecionar regiões de interesse na tela."""
    rois = []
    initial_frame = np.array(ImageGrab.grab(bbox=initial_bbox))
    initial_frame = cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
    for i in range(num_screens):
        bbox = cv2.selectROI(f"Selecione a Região de Interesse (ROI) {i+1} e pressione ENTER", initial_frame, False, False)
        if bbox[2] != 0 and bbox[3] != 0:  # Ignora seleções vazias
            rois.append(bbox)
        cv2.destroyAllWindows()
    return rois
 
def processar_imagens(rois, black_screen, last_class_name, previous_state, time_since_last_update, last_update, cursor):
    """Processar imagens e atualizar banco de dados conforme necessário."""
    tela = ImageGrab.grab(bbox=initial_bbox)
    full_frame = np.array(tela)
    full_frame = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
    black_screen.fill(0)
    for i, bbox in enumerate(rois):
        x, y, w, h = bbox
        roi_frame = full_frame[y:y+h, x:x+w]
        image = cv2.resize(roi_frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image /= 127.5
        image -= 1.0
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
 
        # Desenha os resultados na black_screen
        cv2.putText(black_screen, f"Linha {i+1}: {class_name} - score  {confidence_score:.2f}", (10, 20 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
        #if (last_class_name[i] != class_name) or (last_class_name[i] is None):
        last_class_name[i] = class_name
        timestamp = datetime.now()
        data_to_insert = (timestamp, i+1, class_name, float(confidence_score))
        cursor.execute("INSERT INTO detections (timestamp, roi_number, class_name, confidence_score) VALUES (?, ?, ?, ?)", data_to_insert)
        cursor.commit()
 
        time.sleep(1)
        # Atualiza o estado anterior e o tempo de atualização para cada ROI
        previous_state[i] = class_name
        last_update[i] = time.time()
 
    # Mostrar o frame com as detecções
    cv2.imshow("Analise mesa alimentacao", black_screen)
 
def main():
    num_screens =6  # Definido diretamente para simplificar
    rois = selecionar_rois(num_screens, initial_bbox)
    if not rois:
        print("Nenhuma região selecionada, encerrando.")
        return
 
    with pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}') as cnxn:
        cursor = cnxn.cursor()
        last_class_name = [None] * num_screens
        previous_state = [None] * num_screens
        time_since_last_update = [0] * num_screens
        last_update = [time.time()] * num_screens
        black_screen = np.zeros((400, 400, 3), dtype=np.uint8)
 
        while True:
            processar_imagens(rois, black_screen, last_class_name, previous_state, time_since_last_update, last_update, cursor)
           
            # Força a coleta de lixo
            gc.collect()
           
            # Condição para sair: pressionar Esc
            if cv2.waitKey(1) & 0xFF == 27:
                break
 
        cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
