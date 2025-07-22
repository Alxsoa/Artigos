import cv2
import numpy as np
import random

def aplicar_tremor(frame, intensidade):
    """Aplica um efeito de tremor/tremulação ao frame"""
    h, w = frame.shape[:2]
    
    # Gera deslocamentos aleatórios
    dx = random.randint(-intensidade, intensidade)
    dy = random.randint(-intensidade, intensidade)
    
    # Cria uma matriz de transformação com deslocamento
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Aplica a transformação
    frame_tremido = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return frame_tremido

def processar_video(input_path, output_path, intensidade_tremor=5, mostrar_processamento=True):
    """
    Processa um vídeo adicionando efeito de tremor/tremulação
    
    Parâmetros:
    - input_path: caminho do vídeo de entrada
    - output_path: caminho para salvar o vídeo processado
    - intensidade_tremor: intensidade do efeito de tremor (1-10)
    - mostrar_processamento: se True, mostra o processamento em tempo real
    """
    
    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo de entrada!")
        return
    
    # Obtém propriedades do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define o codec e cria o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ou 'XVID' para AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processando vídeo: {input_path}")
    print(f"Resolução: {width}x{height}, FPS: {fps}")
    print(f"Intensidade do tremor: {intensidade_tremor}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Aplica o efeito de tremor
        frame_tremido = aplicar_tremor(frame, intensidade_tremor)
        
        # Escreve o frame no vídeo de saída
        out.write(frame_tremido)
        
        # Mostra o processamento se habilitado
        if mostrar_processamento:
            cv2.imshow('Vídeo com Tremor', frame_tremido)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Libera recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processamento concluído! Vídeo salvo em: {output_path}")

# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelos caminhos desejados
    video_entrada = r'C:\Users\alexa\Atividades\LocalEstabilizaVídeos\Dataset\VideoBase.mp4'
    video_saida = r'C:\Users\alexa\Atividades\LocalEstabilizaVídeos\Dataset\VideoBaseOUT.mp4'
    # Intensidade do tremor (1-10, sendo 10 mais intenso)
    intensidade = 8
    
    processar_video(video_entrada, video_saida, intensidade)