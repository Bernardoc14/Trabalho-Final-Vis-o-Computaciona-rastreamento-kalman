"""main.py: liga o detector (Pessoa A) ao filtro de Kalman (Pessoa B).

Uso (sempre a partir da raiz do projeto):
    python main.py --video videos/normal.mp4
    python main.py --video videos/oclusao.mp4 --nome oclusao --sd 3 --sm 8
Teclas na janela: q = sair.  Sem janela (so grava): --sem-janela

Gera em resultados/:
    <nome>_saida.mp4          video com as sobreposicoes
    <nome>_deteccoes.csv      frame,zx,zy   (vazio quando nao detectou) -> entrada do experimentos.py
    <nome>_rastreamento.csv   frame,estado,pred_x,pred_y,est_x,est_y
"""
import argparse, csv, os, time
import cv2
from detector import detectar
from kalman import Rastreador

AZUL, VERMELHO, LARANJA = (255, 0, 0), (0, 0, 255), (0, 165, 255)   # BGR
CINZA = (190, 190, 190)
AVISOS = {"no_escuro": "SEM MEDICAO: filtro no escuro",
          "rejeitado": "MEDICAO REJEITADA (fora da regiao de busca)",
          "perdido": "TRILHA PERDIDA"}


def ponto(p):
    return (int(round(p[0])), int(round(p[1])))


def desenhar(frame, k, z, s, trilha, escuro):
    if s is None:                                           # ainda sem trilha
        cv2.putText(frame, "AGUARDANDO DETECCAO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, VERMELHO, 2)
    else:
        if s["regiao"] is not None:                         # elipse de busca (vem da covariancia)
            a, b, ang = s["regiao"]
            eixos = (int(min(a, 2000)) + 1, int(min(b, 2000)) + 1)
            cv2.ellipse(frame, ponto(s["pred"]), eixos, ang, 0, 360, CINZA, 1)
        for i in range(1, len(trilha)):                     # trajetoria estimada
            cv2.line(frame, ponto(trilha[i - 1]), ponto(trilha[i]), LARANJA, 2)
        if z is not None:
            cv2.circle(frame, ponto(z), 6, AZUL, -1)            # medicao (azul)
        cv2.circle(frame, ponto(s["pred"]), 9, VERMELHO, 2)     # predicao (vermelho)
        if s["est"] is not None:
            cv2.circle(frame, ponto(s["est"]), 5, LARANJA, -1)  # estimativa (laranja)
        if s["estado"] != "medido":
            texto = AVISOS[s["estado"]] + (f"  ({escuro} quadros)" if s["estado"] != "perdido" else "")
            cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
            cv2.putText(frame, texto, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # legenda e numero do quadro
    for i, (nome, cor) in enumerate([("medicao", AZUL), ("predicao", VERMELHO), ("estimativa", LARANJA)]):
        cv2.circle(frame, (15 + 130 * i, 465), 6, cor, -1)
        cv2.putText(frame, nome, (28 + 130 * i, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, f"quadro {k}", (540, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/normal.mp4")
    ap.add_argument("--nome", help="prefixo dos arquivos de saida (padrao: nome do video)")
    ap.add_argument("--pasta", default="resultados")
    ap.add_argument("--sd", type=float, default=3.0, help="desvio do ruido do modelo (Sd)")
    ap.add_argument("--sm", type=float, default=8.0, help="desvio do ruido da medicao (Sm), em pixels")
    ap.add_argument("--max-sem", type=int, default=30, help="max. de quadros so com predicao")
    ap.add_argument("--sem-janela", action="store_true")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("Nao consegui abrir:", a.video)
        return
    os.makedirs(a.pasta, exist_ok=True)
    nome = a.nome or os.path.splitext(os.path.basename(a.video))[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    saida = cv2.VideoWriter(f"{a.pasta}/{nome}_saida.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

    rastreador = Rastreador(sd=a.sd, sm=a.sm, max_sem_medicao=a.max_sem)
    det, rast, trilha = [], [], []
    escuro, n_det, n_escuro, k, t0 = 0, 0, 0, 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480))
        z = detectar(frame)                    # (px, py) ou None   <- Pessoa A
        s = rastreador.step(z)                 # predicao + gating + correcao   <- Pessoa B

        n_det += z is not None
        if s is not None and s["estado"] != "medido":
            escuro += 1; n_escuro += 1
        else:
            escuro = 0
        if s is not None and s["est"] is not None:
            trilha.append(s["est"])
        elif s is not None:                    # trilha perdida: recomeca o desenho
            trilha = []

        det.append([k, "" if z is None else round(z[0], 2), "" if z is None else round(z[1], 2)])
        if s is not None:
            est = s["est"] if s["est"] is not None else ("", "")
            rast.append([k, s["estado"], round(s["pred"][0], 2), round(s["pred"][1], 2),
                         "" if est[0] == "" else round(est[0], 2), "" if est[1] == "" else round(est[1], 2)])

        desenhar(frame, k, z, s, trilha, escuro)
        saida.write(frame)
        k += 1
        if not a.sem_janela:
            cv2.imshow("rastreamento", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release(); saida.release(); cv2.destroyAllWindows()
    with open(f"{a.pasta}/{nome}_deteccoes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "zx", "zy"]); w.writerows(det)
    with open(f"{a.pasta}/{nome}_rastreamento.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "estado", "pred_x", "pred_y", "est_x", "est_y"]); w.writerows(rast)

    dt = time.time() - t0
    print(f"{k} quadros | detectou em {n_det} ({100 * n_det / max(k, 1):.0f}%) | "
          f"{n_escuro} quadros sem medicao aceita | {k / max(dt, 1e-9):.0f} quadros/s")
    print(f"Arquivos em {a.pasta}/: {nome}_saida.mp4, {nome}_deteccoes.csv, {nome}_rastreamento.csv")


if __name__ == "__main__":
    main()
