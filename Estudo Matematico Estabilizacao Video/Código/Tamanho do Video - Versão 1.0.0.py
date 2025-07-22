import cv2

def obter_tamanho_video(caminho_video):
    # Abre o vídeo
    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return None
    
    # Obtém as propriedades do vídeo
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Informações do vídeo:")
    print(f"Resolução: {largura}x{altura}")
    print(f"FPS: {fps:.2f}")
    print(f"Total de frames: {total_frames}")
    
    # Mostra o primeiro frame para visualização
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Primeiro Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cap.release()
    return largura, altura

# Exemplo de uso
if __name__ == "__main__":
    caminho = "C:\\Users\\alexa\\Atividades\\LocalEstabilizaVídeos\\Dataset\\ImagemNaoEstabilizadaDrone.mp4"  # Substitua pelo caminho do seu vídeo
    tamanho = obter_tamanho_video(caminho)
    
    if tamanho:
        print(f"\nDimensões do vídeo: {tamanho[0]}x{tamanho[1]}")