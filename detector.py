"""
detector.py  (Pessoa A)  -  UM arquivo so.

Funcao principal (acordo com a Pessoa B):
    detectar(frame) -> (px, py) se achou a bola, ou None se nao achou

Teste (mostra as janelas e tem barras para ajustar a cor e AREA_MIN):
    python detector.py      (usa sempre videos/normal.mp4)
    Teclas: q = sair | p = imprime os parametros | s = salva imagens
"""
import os
import cv2
import numpy as np

# ---------------- PARAMETROS (ajuste com as barras) ----------------
H_MIN, H_MAX = 140, 179   # matiz (cor). Rosa fica perto de 160-175
S_MIN, S_MAX = 50, 255    # saturacao (quanto "colorido")
V_MIN, V_MAX = 100, 255   # brilho
AREA_MIN = 300            # contorno menor que isso e ruido
KERNEL = np.ones((7, 7), np.uint8)   # "pincel" da morfologia


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
def testar(fonte):
    global H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX, AREA_MIN

    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print("Nao consegui abrir:", fonte)
        return

    # barras (comecam com os valores de cima)
    cv2.namedWindow("barras")
    for nome, valor, maximo in [("H_MIN", H_MIN, 179), ("H_MAX", H_MAX, 179),
                                ("S_MIN", S_MIN, 255), ("S_MAX", S_MAX, 255),
                                ("V_MIN", V_MIN, 255), ("V_MAX", V_MAX, 255),
                                ("AREA_MIN", AREA_MIN, 10000)]:
        cv2.createTrackbar(nome, "barras", valor, maximo, lambda _: None)

    while True:
        ok, frame = cap.read()
        if not ok:                      # acabou o video: volta ao inicio
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        cv2.putText(saida, f"AREA_MIN: {AREA_MIN} | Maior: {int(area_maior)}",
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
            break
        elif tecla == ord("p"):
            print(f"H_MIN, H_MAX = {H_MIN}, {H_MAX}")
            print(f"S_MIN, S_MAX = {S_MIN}, {S_MAX}")
            print(f"V_MIN, V_MAX = {V_MIN}, {V_MAX}")
            print(f"AREA_MIN = {AREA_MIN}")
        elif tecla == ord("s"):
            os.makedirs("resultados/etapas", exist_ok=True)
            cv2.imwrite("resultados/etapas/original.png", frame)
            cv2.imwrite("resultados/etapas/hsv.png", hsv)
            cv2.imwrite("resultados/etapas/mascara.png", mascara)
            cv2.imwrite("resultados/etapas/resultado.png", saida)
            print("Imagens salvas em resultados/etapas/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # sempre usa videos/normal.mp4 (pasta videos na mesma pasta do detector.py)
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    video = os.path.join(pasta_atual, "videos", "normal.mp4")
    testar(video)
    