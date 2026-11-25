# Rastreamento de objeto colorido com filtro de Kalman

Trabalho final de Visão Computacional (UFPI, 2026.2). Opção 01: rastreamento de uma bola colorida
com oclusões e predição de trajetória. Equipe: _(colocar os nomes da dupla)_

Um detector por cor (HSV + morfologia + centroide) fornece a posição da bola a cada quadro.
O filtro de Kalman, implementado do zero em NumPy, combina essas medições com um modelo de
movimento e segue estimando a posição nos quadros em que o detector falha.

## Instalação

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Arquivos

| Arquivo | O que faz |
|---|---|
| `detector.py` | `detectar(frame)` devolve `(px, py)` ou `None`. Rodando o arquivo direto, abre barras para ajustar a cor. |
| `kalman.py` | Filtro de Kalman: `KalmanCV` (velocidade constante), `KalmanCA` (aceleração constante) e `Rastreador` (predição, gating, correção e falhas de detecção). |
| `main.py` | Liga detector e filtro, grava o vídeo com as sobreposições e os CSV. |
| `marcar_referencia.py` | Você clica no centro da bola em alguns quadros; gera a referência feita à mão. |
| `experimentos.py` | Métricas e gráficos: bruta x filtrada, 3 valores de Sd, oclusão, CV x CA, previsão a k quadros. |
| `validar_kalman.py` | Confere o `kalman.py` (CV e CA) contra o `cv2.KalmanFilter`. |
| `gerar_video_teste.py` | Vídeo sintético só para testar o código. Não usar no relatório. |
| `videos/` | Vídeos gravados pela equipe: `normal`, `normal2`, `brusco`, `oclusao`. |
| `resultados/` | Saídas geradas (vídeos, CSV, gráficos). |

## Fluxo completo

**Rode tudo a partir da raiz do projeto.** Exemplo com o vídeo `brusco`:

```
# 0. (opcional) conferir a segmentação por cor e ajustar H/S/V
python detector.py videos/brusco.mp4

# 1. detector + filtro: grava o vídeo de saída e os CSV
python main.py --video videos/brusco.mp4

# 2. referência à mão: clique no centro da bola a cada 5 quadros (ESPAÇO = próximo, q = salvar e sair)
python marcar_referencia.py videos/brusco.mp4 5

# 3. experimentos: métricas e gráficos
python experimentos.py --nome brusco

# 4. validação do filtro contra o OpenCV
python validar_kalman.py
```

O que cada etapa gera:

| Etapa | Arquivos |
|---|---|
| `main.py` | `resultados/brusco_saida.mp4`, `resultados/brusco_deteccoes.csv` (`frame,zx,zy`; vazio quando não detectou), `resultados/brusco_rastreamento.csv` (`frame,estado,pred_x,pred_y,est_x,est_y`) |
| `marcar_referencia.py` | `resultados/brusco_referencia.csv` (`frame,rx,ry`) |
| `experimentos.py --nome brusco` | `resultados/brusco/metricas.csv`, `previsao_k.csv`, `trajetorias_sd.png`, `px_tempo_oclusao.png`, `previsao_k.png` |

Os experimentos rodam **sem a câmera**: basta o vídeo gravado, que já vem em `videos/`.
Para reproduzir tudo a partir dos vídeos: repita as etapas 1 a 3 para cada vídeo.

### Opções do `main.py`

| Opção | Padrão | Significado |
|---|---|---|
| `--video` | `videos/normal.mp4` | vídeo de entrada |
| `--sd` | 3.0 | desvio do ruído do modelo (Sd). Pequeno = filtro teimoso; grande = nervoso |
| `--sm` | 8.0 | desvio do ruído da medição (Sm), em pixels |
| `--max-sem` | 30 | máximo de quadros seguidos só com predição; passou disso, a trilha é declarada perdida e reinicia na próxima detecção |
| `--modelo` | `cv` | `cv` velocidade constante, `ca` aceleração constante |
| `--k` | 0 | se maior que 0, desenha a previsão k quadros à frente (cruz verde) |
| `--nome`, `--pasta` | nome do vídeo, `resultados` | prefixo e pasta dos arquivos de saída (com `--modelo ca` o prefixo ganha `_ca`) |
| `--sem-janela` | desligado | só grava, sem abrir janela |

Exemplo do modelo de aceleração com previsão a 10 quadros:
`python main.py --video videos/brusco.mp4 --modelo ca --k 10`

### Opções do `experimentos.py`

`--nome <video>` lê `resultados/<video>_deteccoes.csv` e `resultados/<video>_referencia.csv` e grava em `resultados/<video>/`.
Também aceita `--deteccoes`, `--ref` e `--saida` com caminhos explícitos, `--sm`, `--max-sem`, `--ks 5 10 15` (horizontes da previsão)
e `--demo` (dados sintéticos, só para testar o código).

## Modelos

- **CV** (velocidade constante): estado `[px, py, vx, vy]`. `D` soma a velocidade à posição a cada quadro.
- **CA** (aceleração constante): estado `[px, py, vx, vy, ax, ay]`. `D` soma `v·dt + a·dt²/2` à posição e `a·dt` à velocidade.
  Serve para objeto lançado, em que a gravidade é uma aceleração constante no eixo vertical.
- Nos dois, `M` pega só a posição (o detector só mede onde a bola está), `Sd = sd²·I` (no CA a aceleração tem ruído próprio, `sd/10`) e `Sm = sm²·I`.
- Região de busca: elipse da covariância prevista `S = M P⁻ Mᵀ + Sm`; só medições com distância de Mahalanobis² ≤ 9,21 (99%, 2 graus de liberdade) são aceitas.

Estados da saída do `Rastreador`: `medido`, `no_escuro` (sem medição), `rejeitado` (medição fora da região de busca), `perdido`.

## Usar o filtro em outro código

```python
from kalman import Rastreador
rastreador = Rastreador(sd=3.0, sm=8.0, modelo="cv")   # um por vídeo
for frame in video:
    z = detectar(frame)                    # (px, py) ou None
    saida = rastreador.step(z)             # None enquanto não houver trilha
    # saida["pred"] vermelho | saida["est"] laranja | z azul
    # saida["estado"], saida["regiao"] (a, b, ângulo) para cv2.ellipse
    futuro = rastreador.prever(10)         # posição 10 quadros à frente
```

## Observações

- Os arquivos em `resultados/` vêm dos vídeos reais. O que for gerado com `--demo` ou com `gerar_video_teste.py` é sintético e não entra no relatório.
- Os vídeos terminam com a bola fora do quadro, então o último trecho sem detecção não é uma oclusão: o `experimentos.py` avisa na tela
  se cada buraco cai no início, no meio (oclusão) ou no fim do vídeo.