"""
detector.py  (Pessoa A)  -  UM arquivo so.

Funcao principal (acordo com a Pessoa B):
    detectar(frame) -> (px, py) se achou a bola, ou None se nao achou

Teste (mostra as janelas e tem barras para ajustar a cor e AREA_MIN):
    python detector.py                      (roda normal, normal2, brusco e oclusao em sequencia)
    python detector.py videos/oclusao.mp4   (roda so o video indicado)
    Teclas: q = sair | n = proximo video | p = imprime os parametros | s = salva imagens
"""
import os
import sys
import cv2
import numpy as np

# ---------------- PARAMETROS (ajuste com as barras) ----------------
H_MIN, H_MAX = 162, 179   # matiz (cor)
S_MIN, S_MAX = 123, 255   # saturacao (quanto "colorido")
V_MIN, V_MAX = 101, 255   # brilho
AREA_MIN = 544            # contorno menor que isso e ruido
KERNEL = np.ones((7, 7), np.uint8)   # "pincel" da morfologia

# videos usados no teste (pasta "videos" na mesma pasta do detector.py)
VIDEOS = ["normal.mp4", "normal2.mp4", "brusco.mp4", "oclusao.mp4"]


# ---------------- FUNCAO PRINCIPAL ----------------
def detectar(frame):
    # 1) BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2) mascara: pixels dentro da faixa viram branco, o resto preto
    inferior = np.array([H_MIN, S_MIN, V_MIN])
    superior = np.array([H_MAX, S_MAX, V_MAX])
    mascara = cv2.inRange(hsv, inferior, superior)

    # 3) abertura: tira pontinhos de ruido
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL)
    # 4) fechamento: tapa buracos dentro da bola
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL)

    # 5) pega o maior contorno
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if len(contornos) == 0:
        return None
    maior = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(maior) < AREA_MIN:
        return None

    # 6) centroide = media das posicoes dos pixels do contorno
    m = cv2.moments(maior)
    if m["m00"] == 0:
        return None
    px = m["m10"] / m["m00"]
    py = m["m01"] / m["m00"]
    return (px, py)


# ---------------- TESTE (so roda se chamar este arquivo) ----------------
def testar(fontes):
    global H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX, AREA_MIN

    # barras (comecam com os valores de cima)
    cv2.namedWindow("barras")
    for nome, valor, maximo in [("H_MIN", H_MIN, 179), ("H_MAX", H_MAX, 179),
                                ("S_MIN", S_MIN, 255), ("S_MAX", S_MAX, 255),
                                ("V_MIN", V_MIN, 255), ("V_MAX", V_MAX, 255),
                                ("AREA_MIN", AREA_MIN, 10000)]:
        cv2.createTrackbar(nome, "barras", valor, maximo, lambda _: None)

    i = 0                               # indice do video atual
    cap = None
    nome_video = ""
    sair = False

    while not sair:
        # abre o video atual (se ainda nao esta aberto)
        if cap is None:
            fonte = fontes[i % len(fontes)]
            nome_video = os.path.splitext(os.path.basename(fonte))[0]
            cap = cv2.VideoCapture(fonte)
            if not cap.isOpened():
                print("Nao consegui abrir:", fonte)
                cap = None
                i += 1
                if i >= 2 * len(fontes):    # nenhum video abriu
                    break
                continue
            print("Video:", fonte)

        ok, frame = cap.read()
        if not ok:                      # acabou o video: vai para o proximo
            cap.release()
            cap = None
            i += 1
            continue
        frame = cv2.resize(frame, (640, 480))

        # le as barras e atualiza os parametros
        H_MIN = cv2.getTrackbarPos("H_MIN", "barras")
        H_MAX = cv2.getTrackbarPos("H_MAX", "barras")
        S_MIN = cv2.getTrackbarPos("S_MIN", "barras")
        S_MAX = cv2.getTrackbarPos("S_MAX", "barras")
        V_MIN = cv2.getTrackbarPos("V_MIN", "barras")
        V_MAX = cv2.getTrackbarPos("V_MAX", "barras")
        AREA_MIN = cv2.getTrackbarPos("AREA_MIN", "barras")

        # usa a mesma funcao do projeto
        ponto = detectar(frame)

        # mascara pos-morfologia para conferir exatamente o que entra nos contornos
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, np.array([H_MIN, S_MIN, V_MIN]),
                              np.array([H_MAX, S_MAX, V_MAX]))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        area_maior = cv2.contourArea(max(contornos, key=cv2.contourArea)) if contornos else 0.0

        saida = frame.copy()
        cv2.putText(saida, f"{nome_video} | AREA_MIN: {AREA_MIN} | Maior: {int(area_maior)}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if ponto is not None:
            cv2.circle(saida, (int(ponto[0]), int(ponto[1])), 6, (255, 0, 0), -1)
        else:
            cv2.putText(saida, "SEM DETECCAO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("mascara", mascara)
        cv2.imshow("resultado", saida)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord("q"):
            sair = True
        elif tecla == ord("n"):         # pula para o proximo video
            cap.release()
            cap = None
            i += 1
        elif tecla == ord("p"):
            print(f"H_MIN, H_MAX = {H_MIN}, {H_MAX}")
            print(f"S_MIN, S_MAX = {S_MIN}, {S_MAX}")
            print(f"V_MIN, V_MAX = {V_MIN}, {V_MAX}")
            print(f"AREA_MIN = {AREA_MIN}")
        elif tecla == ord("s"):
            os.makedirs("resultados/etapas", exist_ok=True)
            cv2.imwrite(f"resultados/etapas/{nome_video}_original.png", frame)
            cv2.imwrite(f"resultados/etapas/{nome_video}_hsv.png", hsv)
            cv2.imwrite(f"resultados/etapas/{nome_video}_mascara.png", mascara)
            cv2.imwrite(f"resultados/etapas/{nome_video}_resultado.png", saida)
            print("Imagens salvas em resultados/etapas/")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:               # video(s) passado(s) na linha de comando
        fontes = sys.argv[1:]
    else:                               # padrao: os 4 videos da pasta videos
        fontes = [os.path.join(pasta_atual, "videos", v) for v in VIDEOS]
    testar(fontes)
    