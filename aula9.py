import cv2
import numpy as np

def criar_imagem_simulada():
    """Gera uma imagem simulada com um 'carro', uma 'pessoa' e um 'animal'."""
    # Fundo branco
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Simulando um 'Carro' (Retângulo azul horizontal)
    cv2.rectangle(img, (50, 200), (200, 280), (255, 0, 0), -1)
    
    # Simulando uma 'Pessoa' (Retângulo verde vertical)
    cv2.rectangle(img, (300, 100), (360, 320), (0, 255, 0), -1)
    
    # Simulando um 'Animal' (Retângulo vermelho pequeno/quadrado)
    cv2.rectangle(img, (450, 250), (510, 300), (0, 0, 255), -1)
    
    return img

def simular_detector_objetos(img):
    resultado = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Definição dos intervalos de cores para detecção (Simulação)
    filtros = {
        "Carro": (cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255])), (255, 0, 0)),
        "Pessoa": (cv2.inRange(hsv, np.array([40, 150, 0]), np.array([70, 255, 255])), (0, 255, 0)),
        "Animal": (cv2.inRange(hsv, np.array([0, 150, 0]), np.array([10, 255, 255])), (0, 0, 255))
    }
    
    for classe, (mascara, cor_box) in filtros.items():
        # Encontra contornos da cor específica
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 500: # Filtro para evitar ruídos pequenos
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = float(w) / h
                
                # REGRAS HEURÍSTICAS (Simulação do modelo)
                if classe == "Carro" and aspect_ratio > 1.2:
                    label = f"{classe} (OK)"
                elif classe == "Pessoa" and aspect_ratio < 0.8:
                    label = f"{classe} (OK)"
                elif classe == "Animal" and 0.7 <= aspect_ratio <= 1.3:
                    label = f"{classe} (OK)"
                else:
                    label = "Desconhecido"
                
                # Desenha o resultado se validado pelas regras
                if label != "Desconhecido":
                    cv2.rectangle(resultado, (x, y), (x + w, y + h), cor_box, 2)
                    cv2.putText(resultado, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_box, 2)
                    
    return resultado

if __name__ == "__main__":
    # 1. Pipeline de Execução
    imagem_teste = criar_imagem_simulada()
    imagem_processada = simular_detector_objetos(imagem_teste)
    
    # 2. Salvando o resultado em disco
    cv2.imwrite("resultado_simulacao.png", imagem_processada)
    print("Simulação concluída com sucesso! Imagem 'resultado_simulacao.png' gerada.")