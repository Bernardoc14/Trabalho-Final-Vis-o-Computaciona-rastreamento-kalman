"""Gera um video SINTETICO para testar o pipeline enquanto nao ha videos reais.

    python gerar_video_teste.py
Cria:  videos/teste_sintetico.mp4  (bola rosa passando atras de um "livro" cinza)
       resultados/teste_referencia.csv  (posicao real a cada 10 quadros)
NAO use estes arquivos no relatorio: sao so para testar o codigo.
"""
import csv, os
import cv2
import numpy as np

os.makedirs("videos", exist_ok=True); os.makedirs("resultados", exist_ok=True)
rng = np.random.default_rng(0)
rosa = cv2.cvtColor(np.uint8([[[165, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
out = cv2.VideoWriter("videos/teste_sintetico.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
ref = []
for k in range(130):
    cx, cy = 60 + 4.2 * k, 240 + 90 * np.sin(0.08 * k)
    frame = np.full((480, 640, 3), 245, np.uint8)
    cv2.circle(frame, (int(cx), int(cy)), 18, rosa, -1)
    cv2.rectangle(frame, (300, 110), (380, 370), (110, 110, 110), -1)     # o "livro" (oclusao)
    frame = np.clip(frame + rng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
    out.write(frame)
    if k % 10 == 0:
        ref.append([k, round(cx, 1), round(cy, 1)])
out.release()
with open("resultados/teste_referencia.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "rx", "ry"]); w.writerows(ref)
print("Criados: videos/teste_sintetico.mp4 e resultados/teste_referencia.csv")
