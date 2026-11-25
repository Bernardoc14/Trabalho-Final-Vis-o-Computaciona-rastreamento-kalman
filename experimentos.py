"""Experimentos: bruta x filtrada, 3 valores de Sd, oclusão, CV x CA e previsão a k quadros.

Uso (sempre a partir da raiz do projeto), depois de rodar o main.py e o marcar_referencia.py:
    python experimentos.py --nome brusco
        lê   resultados/brusco_deteccoes.csv  e  resultados/brusco_referencia.csv
        grava em resultados/brusco/
Sem os dados ainda (teste com trajetória sintética):
    python experimentos.py --demo
Caminhos explícitos (opcional):
    python experimentos.py --deteccoes A.csv --ref B.csv --saida pasta

deteccoes.csv:  frame,zx,zy   (zx e zy VAZIOS quando o detector não achou o objeto)
referencia.csv: frame,rx,ry   (posição real marcada à mão em alguns quadros)

Saídas: metricas.csv, previsao_k.csv, trajetorias_sd.png, px_tempo_oclusao.png, previsao_k.png
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kalman import Rastreador

CONFIGS = [("teimoso (elefante)", 0.5), ("intermediário", 3.0), ("nervoso (peixinho-dourado)", 20.0)]
MODELOS = [("cv", "velocidade constante"), ("ca", "aceleração constante")]
SD_PREV = 3.0     # Sd usado no gráfico da previsão a k quadros
SM = 8.0          # erro típico do detector, em pixels (ajuste olhando seus dados)
MAX_SEM = 30      # máx. de quadros só com predição antes de declarar a trilha perdida
LIMIAR_PX = 20.0  # erro de até 20 px conta como "acertou" / "recuperou"
KS = [5, 10, 15]  # horizontes da previsão (quadros à frente)


# ---------------------------------------------------------------- leitura
def ler_deteccoes(caminho):
    z = []
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            z.append(None if r["zx"] == "" else (float(r["zx"]), float(r["zy"])))
    return z


def ler_ref(caminho):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        return {int(r["frame"]): (float(r["rx"]), float(r["ry"])) for r in csv.DictReader(f)}


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


# ---------------------------------------------------------------- execução do filtro
def rodar(z, sd, modelo, ks=()):
    """Roda o Rastreador em todos os quadros. Devolve (saídas, previsões a k quadros)."""
    r = Rastreador(sd=sd, sm=SM, max_sem_medicao=MAX_SEM, modelo=modelo)
    saidas, prev = [], []
    for p in z:
        saidas.append(r.step(p))
        prev.append({k: r.prever(k) for k in ks})
    return saidas, prev


def pos(saidas, chave):
    return [None if (s is None or s[chave] is None) else s[chave] for s in saidas]


# ---------------------------------------------------------------- métricas
def erro_medio(pontos, ref, validos=None):
    e = [np.hypot(pontos[k][0] - ref[k][0], pontos[k][1] - ref[k][1])
         for k in ref if k < len(pontos) and pontos[k] is not None
         and (validos is None or k in validos)]
    return float(np.mean(e)) if e else float("nan")


def tremor(pontos, validos):
    """Desvio padrão da aceleração só onde os 3 quadros são válidos."""
    d = [np.array(pontos[i + 2]) - 2 * np.array(pontos[i + 1]) + np.array(pontos[i])
         for i in range(len(pontos) - 2)
         if all(validos[j] and pontos[j] is not None for j in (i, i + 1, i + 2))]
    return float(np.std(d)) if d else float("nan")


def recuperacao(est, estados, ref, i0, i1):
    """Dentro da oclusão: (nº de pontos de referência, quantos com erro <= LIMIAR_PX).
    Depois dela: quantos quadros até o filtro voltar a aceitar medição (None = nunca)."""
    na_oc = [k for k in ref if i0 <= k < i1 and k < len(est) and est[k] is not None]
    ok = sum(1 for k in na_oc
             if np.hypot(est[k][0] - ref[k][0], est[k][1] - ref[k][1]) <= LIMIAR_PX)
    volta = next((k - i1 for k in range(i1, len(estados)) if estados[k] == "medido"), None)
    return len(na_oc), ok, volta


def erro_reentrada(z, saidas, i0, i1):
    """Distância entre a PREDIÇÃO do filtro e a 1ª medição depois da oclusão (quadro i1).
    Não depende de referência marcada à mão (a bola está escondida durante a oclusão, então
    quase não há referência lá dentro). 'perdida' se a trilha foi reiniciada durante a oclusão."""
    if i1 <= i0 or i1 >= len(z) or z[i1] is None or saidas[i1] is None:
        return "n/a"
    if any(s is not None and s["estado"] == "perdido" for s in saidas[i0:i1 + 1]):
        return "perdida"
    return round(float(np.hypot(saidas[i1]["pred"][0] - z[i1][0], saidas[i1]["pred"][1] - z[i1][1])), 2)


def atraso(z, est, kmax=10):
    """k que melhor alinha est[t] com z[t-k]: quantos quadros a estimativa 'atrasa'."""
    melhor, menor = 0, 1e18
    for k in range(kmax + 1):
        d = [np.hypot(est[t][0] - z[t - k][0], est[t][1] - z[t - k][1])
             for t in range(k, len(z)) if est[t] is not None and z[t - k] is not None]
        if len(d) > 10 and np.mean(d) < menor:
            melhor, menor = k, np.mean(d)
    return melhor


def buracos(z):
    """Todos os trechos seguidos sem detecção: lista de (início, tamanho)."""
    out, ini = [], None
    for i, p in enumerate(list(z) + [(0, 0)]):
        if p is None and ini is None:
            ini = i
        if p is not None and ini is not None:
            out.append((ini, i - ini))
            ini = None
    return out


def maior_buraco(z):
    """Maior buraco ENTRE duas detecções (a oclusão). Os buracos no começo (antes da 1ª
    detecção) e no fim (bola saiu do quadro) não contam como oclusão. Devolve (início, fim)."""
    idx = [i for i, p in enumerate(z) if p is not None]
    melhor, ini = (0, 0), None
    for i in range(idx[0], idx[-1] + 1):
        if z[i] is None and ini is None:
            ini = i
        if z[i] is not None and ini is not None:
            if i - ini > melhor[0]:
                melhor = (i - ini, ini)
            ini = None
    return melhor[1], melhor[1] + melhor[0]


def metricas_previsao(z, saidas, prev, ref, k):
    """Previsão a k quadros: a partir do quadro t (com medição), onde a bola estará em t+k?
    Compara com a referência marcada à mão em t+k. A previsão ingênua 'a bola fica onde foi
    medida' (sem filtro) é calculada nos MESMOS casos, para servir de linha de base."""
    e_mod, e_base = [], []
    for f, (rx, ry) in ref.items():
        t = f - k
        if t < 0 or t >= len(z) or z[t] is None or prev[t][k] is None:
            continue
        e_mod.append(np.hypot(prev[t][k][0] - rx, prev[t][k][1] - ry))
        e_base.append(np.hypot(z[t][0] - rx, z[t][1] - ry))

    def resumo(e):
        return (round(float(np.mean(e)), 2), round(100 * float(np.mean(np.array(e) <= LIMIAR_PX)), 1)) \
            if e else (float("nan"), float("nan"))
    return len(e_mod), resumo(e_mod), resumo(e_base)


# ---------------------------------------------------------------- programa principal
def main():
    global SM, MAX_SEM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", help="atalho: lê resultados/<nome>_deteccoes.csv e _referencia.csv, "
                                   "grava em resultados/<nome>/")
    ap.add_argument("--deteccoes"); ap.add_argument("--ref"); ap.add_argument("--saida")
    ap.add_argument("--pasta", default="resultados", help="pasta base usada com --nome")
    ap.add_argument("--demo", action="store_true", help="dados sintéticos (só para testar o código)")
    ap.add_argument("--sm", type=float, default=SM, help="desvio do ruído da medição (px)")
    ap.add_argument("--max-sem", type=int, default=MAX_SEM, help="máx. de quadros só com predição")
    ap.add_argument("--ks", type=int, nargs="+", default=KS, help="horizontes da previsão, em quadros")
    a = ap.parse_args()
    SM, MAX_SEM = a.sm, a.max_sem

    if a.demo:
        z, ref = demo()
        saida = a.saida or os.path.join(a.pasta, "demo")
    else:
        det = a.deteccoes or (f"{a.pasta}/{a.nome}_deteccoes.csv" if a.nome else None)
        rf = a.ref or (f"{a.pasta}/{a.nome}_referencia.csv" if a.nome else None)
        if not det or not rf:
            ap.error("use --nome <video>, ou --deteccoes e --ref, ou --demo")
        z, ref = ler_deteccoes(det), ler_ref(rf)
        saida = a.saida or (f"{a.pasta}/{a.nome}" if a.nome else a.pasta)
    os.makedirs(saida, exist_ok=True)

    # --- buracos de detecção (para a discussão no relatório)
    N = len(z)
    for ini, n in buracos(z):
        if n >= 5:
            onde = "início (antes da 1ª detecção)" if ini == 0 else \
                   "fim (bola saiu do quadro?)" if ini + n == N else "meio (oclusão)"
            print(f"Buraco de {n} quadros sem detecção: quadros {ini} a {ini + n - 1}  [{onde}]")
    i0, i1 = maior_buraco(z)
    # quadros comparaveis: detector achou e fora da oclusao (+5 quadros de convergencia)
    validos = [p is not None and not (i0 <= k < i1 + 5) for k, p in enumerate(z)]
    comuns = {k for k in ref if k < N and validos[k]}
    # Sm estimado: desvio por eixo entre detecção e referência (inclui o erro do clique manual,
    # então é um limite superior do erro real do detector)
    d2 = [(z[k][0] - ref[k][0]) ** 2 + (z[k][1] - ref[k][1]) ** 2 for k in comuns]
    sm_est = float(np.sqrt(np.mean(d2) / 2)) if d2 else float("nan")
    print(f"Oclusão (maior buraco entre detecções): quadros {i0} a {i1 - 1} ({i1 - i0} quadros)")
    print(f"Sm estimado a partir dos dados: {sm_est:.1f} px (valor usado no filtro: {SM})")
    print(f"Pontos de referência: {len(ref)} no total, {len([k for k in ref if i0 <= k < i1])} dentro da oclusão\n")

    # --- roda tudo: 2 modelos x 3 valores de Sd
    linhas, linhas_k, runs = [], [], {}
    for modelo, _ in MODELOS:
        for nome, sd in CONFIGS:
            s, prev = rodar(z, sd, modelo, a.ks)
            runs[(modelo, sd)] = s
            est = pos(s, "est")
            estados = [x["estado"] if x else None for x in s]
            n_ref, n_ok, volta = recuperacao(est, estados, ref, i0, i1)
            linhas.append(dict(
                modelo=modelo, config=nome, sd=sd,
                erro_bruto_px=round(erro_medio(z, ref, comuns), 2),
                erro_filtrado_px=round(erro_medio(est, ref, comuns), 2),
                tremor_bruto=round(tremor(z, validos), 2), tremor_filtrado=round(tremor(est, validos), 2),
                atraso_quadros=atraso(z, est),
                oclusao_quadros=i1 - i0,
                recuperados_ate_20px=f"{n_ok}/{n_ref}",
                quadros_ate_retomar="nunca" if volta is None else volta,
                erro_na_oclusao_px=round(erro_medio([est[k] if i0 <= k < i1 else None for k in range(len(est))], ref), 2),
                erro_reentrada_px=erro_reentrada(z, s, i0, i1),
                rejeitados=estados.count("rejeitado"),
                trilhas_perdidas=estados.count("perdido"),
            ))
            for k in a.ks:
                n, (em, am), (eb, ab) = metricas_previsao(z, s, prev, ref, k)
                linhas_k.append(dict(modelo=modelo, sd=sd, k=k, n_casos=n,
                                     erro_modelo_px=em, acerto_modelo_pct=am,
                                     erro_ultima_medicao_px=eb, acerto_ultima_medicao_pct=ab))

    # --- gráfico 1: trajetória bruta x filtrada, 3 Sd, nos dois modelos
    zz = np.array([p for p in z if p is not None])
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for i, (modelo, nome_modelo) in enumerate(MODELOS):
        for (nome, sd), eixo in zip(CONFIGS, ax[i]):
            ee = np.array([p for p in pos(runs[(modelo, sd)], "est") if p is not None])
            eixo.scatter(zz[:, 0], zz[:, 1], s=8, c="tab:blue", alpha=.5, label="medição (bruta)")
            eixo.plot(ee[:, 0], ee[:, 1], c="tab:orange", lw=1.6, label="estimativa (filtrada)")
            eixo.set_title(f"{modelo.upper()} | Sd = {sd}  ({nome})", fontsize=10)
            eixo.grid(alpha=.3)
        ax[i][0].set_ylabel("py")
    for eixo in ax[1]:
        eixo.set_xlabel("px")
    # enquadra só a região das medições: a predição no escuro pode sair do quadro e esticar os eixos
    mx, my = 0.1 * np.ptp(zz[:, 0]) + 10, 0.1 * np.ptp(zz[:, 1]) + 10
    ax[0][0].set_xlim(zz[:, 0].min() - mx, zz[:, 0].max() + mx)
    ax[0][0].set_ylim(zz[:, 1].max() + my, zz[:, 1].min() - my)      # y invertido (imagem)
    ax[0][0].legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{saida}/trajetorias_sd.png", dpi=140); plt.close()

    # --- gráfico 2: px x tempo (Sd intermediário), CV x CA, destacando a oclusão
    plt.figure(figsize=(11, 4.5))
    plt.scatter([k for k, p in enumerate(z) if p], [p[0] for p in z if p], s=8, c="tab:blue", label="medição")
    if ref:
        plt.scatter(list(ref), [v[0] for v in ref.values()], s=26, marker="x", c="tab:green",
                    label="referência (à mão)", zorder=5)
    for modelo, cor in (("cv", "tab:orange"), ("ca", "tab:red")):
        est = pos(runs[(modelo, SD_PREV)], "est")
        plt.plot([k for k, p in enumerate(est) if p], [p[0] for p in est if p], c=cor, lw=1.4,
                 label=f"estimativa {modelo.upper()} (Sd={SD_PREV})")
    plt.axvspan(i0, i1, color="gray", alpha=.2, label="oclusão (sem medição)")
    mpx = 0.1 * np.ptp(zz[:, 0]) + 10
    plt.ylim(zz[:, 0].min() - mpx, zz[:, 0].max() + mpx)     # a predição fora do quadro não estica o eixo
    plt.xlabel("quadro"); plt.ylabel("px"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{saida}/px_tempo_oclusao.png", dpi=140); plt.close()

    # --- gráfico 3: previsão a k quadros (Sd intermediário): erro e acerto
    def linha_k(modelo, k):
        return next(l for l in linhas_k if l["modelo"] == modelo and l["sd"] == SD_PREV and l["k"] == k)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for chave, rotulo, cor in (("erro_modelo_px", "CV", "tab:orange"), ("erro_modelo_px", "CA", "tab:red")):
        m = rotulo.lower()
        ax[0].plot(a.ks, [linha_k(m, k)["erro_modelo_px"] for k in a.ks], "o-", c=cor, label=rotulo)
        ax[1].plot(a.ks, [linha_k(m, k)["acerto_modelo_pct"] for k in a.ks], "o-", c=cor, label=rotulo)
    ax[0].plot(a.ks, [linha_k("cv", k)["erro_ultima_medicao_px"] for k in a.ks], "o--", c="gray",
               label="última medição (sem filtro)")
    ax[1].plot(a.ks, [linha_k("cv", k)["acerto_ultima_medicao_pct"] for k in a.ks], "o--", c="gray",
               label="última medição (sem filtro)")
    ax[0].set_ylabel("erro médio (px)"); ax[1].set_ylabel(f"acerto (erro ≤ {LIMIAR_PX:.0f} px, %)")
    for e in ax:
        e.set_xlabel("k (quadros à frente)"); e.set_xticks(a.ks); e.grid(alpha=.3)
    ax[0].legend(fontsize=8); plt.suptitle(f"Previsão a k quadros (Sd = {SD_PREV})")
    plt.tight_layout(); plt.savefig(f"{saida}/previsao_k.png", dpi=140); plt.close()

    # --- tabelas (utf-8-sig: o Excel abre com os acentos certos)
    with open(f"{saida}/metricas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=linhas[0].keys()); w.writeheader(); w.writerows(linhas)
    with open(f"{saida}/previsao_k.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=linhas_k[0].keys()); w.writeheader(); w.writerows(linhas_k)

    print("MÉTRICAS (modelo, Sd):")
    for l in linhas:
        print(" ", l)
    print("\nPREVISÃO A k QUADROS:")
    for l in linhas_k:
        print(" ", l)
    print(f"\nArquivos em {os.path.abspath(saida)}: metricas.csv, previsao_k.csv, "
          "trajetorias_sd.png, px_tempo_oclusao.png, previsao_k.png")


if __name__ == "__main__":
    main()