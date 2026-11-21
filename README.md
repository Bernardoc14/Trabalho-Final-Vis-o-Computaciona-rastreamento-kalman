# Trabalho-Final-Vis-o-Computaciona-rastreamento-kalman
Rastreamento de objeto colorido com filtro de Kalman (Visão Computacional, UFPI)



# Parte B: Filtro de Kalman e experimentos

## Arquivos (na raiz do projeto)
- `kalman.py`: classe `KalmanCV` (predict, update, gating) e `Rastreador` (ciclo completo + falhas).
- `validar_kalman.py`: compara com `cv2.KalmanFilter` (diferença esperada ~1e-13).
- `experimentos.py`: 3 valores de Sd, métricas e gráficos.

## Como o colega (parte A) usa o filtro
```python
from kalman import Rastreador
rastreador = Rastreador(sd=3.0, sm=8.0)      # a cada vídeo
for frame in video:
    z = detectar(frame)                      # (px, py) ou None
    saida = rastreador.step(z)               # None enquanto não houver trilha
    # saida["pred"] -> vermelho | saida["est"] -> laranja | z -> azul
    # saida["estado"] in {"medido","no_escuro","rejeitado","perdido"}
    # saida["regiao"] -> (semi-eixo a, semi-eixo b, ângulo) para cv2.ellipse
```
Mostrar o aviso "sem medição" quando `estado` for `no_escuro`, `rejeitado` ou `perdido`.

## Formato dos CSV para os experimentos
- `resultados/deteccoes.csv`: `frame,zx,zy` (zx e zy vazios quando não detectou).
- `resultados/referencia.csv`: `frame,rx,ry` (posição real marcada à mão em ~20 quadros).

## Rodar (sempre a partir da raiz do projeto)
```
python validar_kalman.py
python experimentos.py --demo --saida resultados   # teste com dados sintéticos
python experimentos.py --deteccoes resultados/deteccoes.csv --ref resultados/referencia.csv --saida resultados
```
Os arquivos de `resultados/` gerados com `--demo` são só um exemplo (dados sintéticos).
Não use no relatório: gere de novo com os dados reais.