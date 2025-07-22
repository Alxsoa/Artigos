import cv2
import numpy as np

def processar_video(entrada, saida, total_frames_desejado=1362):
    # Abrir vídeo de entrada
    cap = cv2.VideoCapture(entrada)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo de entrada")
        return False

    # Obter propriedades originais
    largura_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Informações do vídeo original:")
    print(f"Resolução: {largura_original}x{altura_original}")
    print(f"FPS: {fps_original:.2f}")
    print(f"Total de frames: {total_frames_original}\n")

    # Configurações desejadas
    LARGURA = 720
    ALTURA = 480
    FPS = 29.15

    # Calcular fator de amostragem para obter exatamente 1362 frames
    fator_amostragem = total_frames_original / total_frames_desejado

    # Criar objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(saida, fourcc, FPS, (LARGURA, ALTURA))

    frame_count = 0
    frames_processados = 0

    while frames_processados < total_frames_desejado:
        # Calcular qual frame pegar (amostragem uniforme)
        frame_pos = int(frames_processados * fator_amostragem)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar frame para 720x480
        frame_redimensionado = cv2.resize(frame, (LARGURA, ALTURA), interpolation=cv2.INTER_AREA)
        
        # Escrever frame no novo vídeo
        out.write(frame_redimensionado)
        frames_processados += 1

        # Mostrar progresso
        if frames_processados % 100 == 0:
            print(f"Processando frame {frames_processados}/{total_frames_desejado}")

    # Liberar recursos
    cap.release()
    out.release()

    print("\nProcessamento concluído!")
    print(f"Vídeo de saída criado com:")
    print(f"Resolução: {LARGURA}x{ALTURA}")
    print(f"FPS: {FPS:.2f}")
    print(f"Total de frames: {frames_processados}")
    
    # Verificar se atingimos o número desejado
    if frames_processados == total_frames_desejado:
        return True
    else:
        print(f"AVISO: Foram processados {frames_processados} frames em vez de {total_frames_desejado}")
        return False

if __name__ == "__main__":
    video_entrada = r"C:\Users\alexa\Atividades\LocalEstabilizaVídeos\Dataset\VideoBaseOUT.mp4"
    video_saida   = r"C:\Users\alexa\Atividades\LocalEstabilizaVídeos\Dataset\VideoBasePadronizado.mp4"

    if processar_video(video_entrada, video_saida, 313):
        print(f"\nVídeo processado com sucesso e salvo como '{video_saida}'")
        print("O vídeo de saída contém exatamente 1362 frames.")
    else:
        print("Ocorreu um erro no processamento do vídeo")

