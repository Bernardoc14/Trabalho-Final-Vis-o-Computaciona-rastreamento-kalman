import cv2, csv, os, sys
from main import ajustar          # mesmas coordenadas do detector (640x360)

video = sys.argv[1]
passo = int(sys.argv[2]) if len(sys.argv) > 2 else 5
nome = os.path.splitext(os.path.basename(video))[0]
cap, refs, clique, k = cv2.VideoCapture(video), [], {}, 0

def cb(ev, x, y, fl, pa):
    if ev == cv2.EVENT_LBUTTONDOWN:
        clique["p"] = (x, y)

def salvar():
    os.makedirs("resultados", exist_ok=True)
    with open(f"resultados/{nome}_referencia.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "rx", "ry"]); w.writerows(refs)
    print(len(refs), "pontos salvos")

cv2.namedWindow("ref"); cv2.setMouseCallback("ref", cb)
while True:
    ok, frame = cap.read()
    if not ok: break
    if k % passo == 0:
        frame = ajustar(frame); clique.clear()
        while True:
            vis = frame.copy()
            if "p" in clique: cv2.circle(vis, clique["p"], 5, (0, 255, 0), -1)
            cv2.putText(vis, f"quadro {k}: clique no centro da bola | ESPACO = proximo | q = sair",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("ref", vis)
            t = cv2.waitKey(20) & 0xFF
            if t == 32: break
            if t == ord("q"): salvar(); sys.exit()
        if "p" in clique: refs.append([k, *clique["p"]])
    k += 1
salvar()