"""experimentos mínimos (bruta x filtrada, 3 valores de Sd, oclusão).

Com os dados reais:
    python experimentos.py --deteccoes ../resultados/deteccoes.csv --ref ../resultados/referencia.csv
Sem os dados ainda (teste com trajetória sintética):
    python experimentos.py --demo

deteccoes.csv:  frame,zx,zy   (zx e zy VAZIOS quando o detector não achou o objeto)
referencia.csv: frame,rx,ry   (posição real marcada à mão em ~20 quadros)
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kalman import Rastreador

CONFIGS = [("teimoso (elefante)", 0.5), ("intermediário", 3.0), ("nervoso (peixinho-dourado)", 20.0)]
SM = 8.0  # erro típico do detector, em pixels (ajuste olhando seus dados)


def ler_deteccoes(caminho):
    z = []
    for r in csv.DictReader(open(caminho)):
        z.append(None if r["zx"] == "" else (float(r["zx"]), float(r["zy"])))
    return z


def ler_ref(caminho):
    return {int(r["frame"]): (float(r["rx"]), float(r["ry"])) for r in csv.DictReader(open(caminho))}


def demo():
    rng = np.random.default_rng(1)
    N = 200
    t = np.arange(N)
    v = np.c_[320 + 250 * np.sin(0.03 * t), 240 + 120 * np.sin(0.07 * t)]
    z = [tuple(p + rng.normal(0, 6, 2)) for p in v]
    for k in rng.choice(N, 10, replace=False):
        z[k] = None                                   # falhas aleatórias
    for k in range(80, 105):
        z[k] = None                                   # oclusão de 25 quadros
    for k in (45, 135, 155):
        z[k] = tuple(np.array(z[k]) + 150)            # medições espúrias
    ref = {k: tuple(v[k] + rng.normal(0, 1, 2)) for k in range(0, N, 10)}
    return z, ref


def rodar(z, sd):
    r = Rastreador(sd=sd, sm=SM)
    return [r.step(p) for p in z]


def pos(saidas, chave):
    return [None if (s is None or s[chave] is None) else s[chave] for s in saidas]


def erro_medio(pontos, ref):
    e = [np.hypot(pontos[k][0] - ref[k][0], pontos[k][1] - ref[k][1])
         for k in ref if k < len(pontos) and pontos[k] is not None]
    return float(np.mean(e)) if e else float("nan")


def tremor(pontos):
    """Desvio padrão da aceleração (2ª diferença) em trechos contínuos. Menor = mais suave."""
    d = [np.array(pontos[i + 2]) - 2 * np.array(pontos[i + 1]) + np.array(pontos[i])
         for i in range(len(pontos) - 2) if None not in (pontos[i], pontos[i + 1], pontos[i + 2])]
    return float(np.std(d)) if d else float("nan")


def atraso(z, est, kmax=10):
    """k que melhor alinha est[t] com z[t-k]: quantos quadros a estimativa 'atrasa'."""
    melhor, menor = 0, 1e18
    for k in range(kmax + 1):
        d = [np.hypot(est[t][0] - z[t - k][0], est[t][1] - z[t - k][1])
             for t in range(k, len(z)) if est[t] is not None and z[t - k] is not None]
        if len(d) > 10 and np.mean(d) < menor:
            melhor, menor = k, np.mean(d)
    return melhor


def maior_buraco(z):
    """Maior sequência de quadros sem medição: (início, fim exclusivo)."""
    melhor, ini = (0, 0), None
    for i, p in enumerate(z + [(0, 0)]):
        if p is None and ini is None:
            ini = i
        if p is not None and ini is not None:
            melhor = max(melhor, (i - ini, ini)) if (i - ini) > (melhor[0]) else melhor
            ini = None
    n, i0 = melhor
    return i0, i0 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deteccoes"); ap.add_argument("--ref")
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--saida", default="../resultados")
    a = ap.parse_args()
    z, ref = demo() if a.demo else (ler_deteccoes(a.deteccoes), ler_ref(a.ref))
    os.makedirs(a.saida, exist_ok=True)
    i0, i1 = maior_buraco(z)
    linhas = []

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for (nome, sd), eixo in zip(CONFIGS, ax):
        s = rodar(z, sd)
        est, pred_ = pos(s, "est"), pos(s, "pred")
        estados = [x["estado"] if x else None for x in s]
        # --- métricas ---
        oc = range(i0, i1)
        prevendo = sum(1 for k in oc if est[k] is not None and estados[k] in ("no_escuro", "rejeitado"))
        reacq = i1 < len(s) and estados[i1] == "medido"
        linhas.append(dict(
            config=nome, sd=sd,
            erro_bruto_px=round(erro_medio(z, ref), 2), erro_filtrado_px=round(erro_medio(est, ref), 2),
            tremor_bruto=round(tremor(z), 2), tremor_filtrado=round(tremor(est), 2),
            atraso_quadros=atraso(z, est),
            oclusao_quadros=i1 - i0, quadros_recuperados=prevendo,
            retomou_apos_oclusao="sim" if reacq else "não",
            erro_na_oclusao_px=round(erro_medio([est[k] if i0 <= k < i1 else None for k in range(len(est))], ref), 2),
        ))
        # --- gráfico ---
        zz = np.array([p for p in z if p is not None])
        ee = np.array([p for p in est if p is not None])
        eixo.scatter(zz[:, 0], zz[:, 1], s=8, c="tab:blue", alpha=.5, label="medição (bruta)")
        eixo.plot(ee[:, 0], ee[:, 1], c="tab:orange", lw=1.6, label="estimativa (filtrada)")
        eixo.set_title(f"Sd = {sd}  ({nome})"); eixo.set_xlabel("px"); eixo.grid(alpha=.3)
    ax[0].set_ylabel("py"); ax[0].invert_yaxis(); ax[0].legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.saida}/trajetorias_sd.png", dpi=140); plt.close()

    # gráfico px x tempo para o Sd intermediário, destacando a oclusão
    s = rodar(z, 3.0); est = pos(s, "est")
    plt.figure(figsize=(10, 4))
    plt.scatter([k for k, p in enumerate(z) if p], [p[0] for p in z if p], s=8, c="tab:blue", label="medição")
    plt.plot([k for k, p in enumerate(est) if p], [p[0] for p in est if p], c="tab:orange", label="estimativa")
    plt.axvspan(i0, i1, color="gray", alpha=.2, label="oclusão (sem medição)")
    plt.xlabel("quadro"); plt.ylabel("px"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{a.saida}/px_tempo_oclusao.png", dpi=140); plt.close()

    with open(f"{a.saida}/metricas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=linhas[0].keys()); w.writeheader(); w.writerows(linhas)
    print(f"Oclusão detectada: quadros {i0} a {i1 - 1} ({i1 - i0} quadros)\n")
    for l in linhas:
        print(l)
    print(f"\nArquivos em {os.path.abspath(a.saida)}: metricas.csv, trajetorias_sd.png, px_tempo_oclusao.png")


if __name__ == "__main__":
    main()
