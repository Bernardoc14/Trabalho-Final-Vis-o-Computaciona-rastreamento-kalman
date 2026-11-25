# Notas para o relatório (números gerados com os CSV que estão no zip)

## 1. Os buracos de 35, 30 e 28 quadros

Os três ficam no **fim** do vídeo, não no meio, então não são oclusões: a bola saiu do quadro (ou ficou fora de vista até o vídeo acabar).

| Vídeo | Quadros | Buraco | Onde | O que o filtro faz (max_sem = 30) |
|---|---|---|---|---|
| brusco | 393 | 35 (quadros 358 a 392) | fim | passa de 30 quadros no escuro, a trilha é declarada **perdida no quadro 388** (1 trilha perdida) |
| oclusao | 312 | 30 (282 a 311) | fim | chega a 30, não passa; trilha **não** é perdida |
| normal2 | 345 | 28 (317 a 344) | fim | 28 < 30; trilha **não** é perdida |

Os três vídeos também começam com 40 a 45 quadros sem detecção (a bola ainda não entrou em cena). O filtro só cria a trilha na primeira detecção.
Texto sugerido: o limite `max_sem = 30` é o comportamento esperado; ele evita que a predição siga indefinidamente para fora do quadro.
No brusco a predição chega a px ≈ 1400 (o quadro tem 640), o que mostra o limite do modelo: **vale mencionar** que uma regra "descartar a trilha quando a predição sai do quadro" resolveria isso mais cedo.

Os buracos do meio (os que o `experimentos.py` usa como oclusão):

- **brusco**, 17 quadros (117 a 133): a bola está em x ≈ 629 indo para a direita em alta velocidade (6 a 16 px/quadro), ou seja, sai pela borda direita (largura 640) e volta em x ≈ 617. É saída do quadro, não oclusão por obstáculo.
- **oclusao**, 27 quadros (141 a 167): antes, a bola está em x ≈ 457 quase parada (≈ −0,3 px/quadro); depois reaparece em x ≈ 313, 144 px à esquerda. Um modelo de velocidade constante prevê ~13 px de deslocamento. O erro de ~130 px na reentrada **não é bug do filtro**: a bola se moveu enquanto estava escondida de um jeito que nenhum dos dois modelos prevê.
  A medição foi aceita na hora (`quadros_ate_retomar = 0`) porque a região de busca cresce com a covariância durante os 27 quadros no escuro.

## 2. Resultados (Sd = 3, sm = 8)

| | brusco CV | brusco CA | oclusao CV | oclusao CA |
|---|---|---|---|---|
| erro médio bruto (px) | 7,29 | 7,29 | 5,56 | 5,56 |
| erro médio filtrado (px) | 7,69 | 7,21 | 5,53 | 5,56 |
| tremor bruto → filtrado | 4,40 → 2,79 | 4,40 → 2,91 | 1,11 → 0,71 | 1,11 → 0,75 |
| erro de reentrada (px) | 152,9 | 44,5 | 130,0 | 140,5 |

Elefante x peixinho-dourado (brusco, CV): com Sd = 0,5 o filtro rejeita 164 medições, perde a trilha 6 vezes e o erro filtrado vai a 139,9 px (teimoso demais para movimento brusco).
Com Sd = 20 o erro é 7,27 px, igual ao bruto (7,29), e o tremor só cai de 4,40 para 4,11: o filtro quase não filtra. Sd = 3 reduz o tremor em 37% sem aumentar o erro.

## 3. Previsão a k quadros (brusco, Sd = 3)

Erro médio em px (e % de acerto com erro ≤ 20 px) contra a referência marcada à mão. "Última medição" = prever que a bola fica onde foi vista, sem filtro.

| k | CV | CA | última medição |
|---|---|---|---|
| 5 | 38,7 (40,0%) | 43,4 (21,7%) | 60,3 (15,0%) |
| 10 | 101,8 (6,9%) | 121,5 (10,3%) | 108,1 (5,2%) |
| 15 | 176,7 (1,8%) | 236,0 (0,0%) | 143,5 (1,8%) |

Com 56 a 60 casos por linha. O CV só ganha da linha de base até k ≈ 10; em k = 15 perde, porque o movimento do `brusco.mp4` muda de direção o tempo todo.
O CA não ajuda aqui. Ele deveria ajudar num lançamento (trajetória parabólica), mas nenhum dos 4 vídeos tem um lançamento.

## 4. O que ainda fraqueja (vale resolver antes de entregar)

1. **Quase não há referência dentro das oclusões** (1 ponto em cada): `recuperados_ate_20px` fica 0/1 ou 1/1, o que não prova nada. Como a bola está escondida, não dá para clicar nela. Por isso acrescentei `erro_reentrada_px` (distância entre a predição e a 1ª medição depois da oclusão), que não depende de clique.
2. **Falta um vídeo com lançamento** para a comparação CV x CA (que a Opção 01 pede) mostrar a diferença. Gravar um `lancamento.mp4` com a bola jogada para cima ou em arco deve fazer o CA vencer o CV no k quadros à frente.
3. **normal2 e normal** ainda não têm `_referencia.csv` nem `experimentos.py` rodado; o zip só tem brusco e oclusao.
4. O `metricas.csv` antigo foi gravado com a codificação padrão do Windows (`intermedi�rio` em outros sistemas). O novo grava em UTF-8 com BOM, que o Excel abre certo.
5. Velocidade: o `main.py` imprime os quadros/s ao final. Anote o valor da sua máquina no relatório (o mínimo exigido é 15 quadros/s com 640x480).
