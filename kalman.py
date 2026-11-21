"""Parte B: filtro de Kalman (velocidade constante) implementado em NumPy.

Estado:   x = [px, py, vx, vy]^T
Medição:  z = [px, py]^T   (o detector só mede a posição)

Uso (o colega do detector só precisa disto):
    rastreador = Rastreador(sd=3.0, sm=8.0)
    saida = rastreador.step(z)   # z = (px, py) ou None
    # saida["pred"] -> vermelho | saida["est"] -> laranja | z -> azul
"""
import numpy as np


class KalmanCV:
    def __init__(self, z0, dt=1.0, sd=3.0, sm=8.0, vel0=20.0):
        # D: transição. Nova posição = posição + velocidade*dt; velocidade não muda.
        self.D = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], float)
        # M: medição. Pega só px e py do estado.
        self.M = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], float)
        # Sd (ruído do modelo): quanto o objeto pode "fugir" do movimento constante.
        # Sd pequeno = confia no modelo (teimoso); grande = aceita mudanças (nervoso).
        self.Sd = (sd ** 2) * np.eye(4)
        # Sm (ruído da medição): erro típico do detector, em pixels.
        self.Sm = (sm ** 2) * np.eye(2)
        # Estado inicial: posição = 1ª medição; velocidade desconhecida (0, mas incerta).
        self.x = np.array([z0[0], z0[1], 0.0, 0.0])
        self.P = np.diag([sm ** 2, sm ** 2, vel0 ** 2, vel0 ** 2])

    def predict(self):
        """x- = D x ;  P- = D P D^T + Sd"""
        self.x = self.D @ self.x
        self.P = self.D @ self.P @ self.D.T + self.Sd
        return self.x[:2].copy()

    def S(self):
        """Covariância da medição esperada: S = M P- M^T + Sm (usada no gating)."""
        return self.M @ self.P @ self.M.T + self.Sm

    def update(self, z):
        """K = P- M^T S^-1 ;  x = x- + K (z - M x-) ;  P = (I - K M) P-"""
        y = np.asarray(z, float) - self.M @ self.x      # inovação (erro da predição)
        K = self.P @ self.M.T @ np.linalg.inv(self.S())  # ganho de Kalman
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.M) @ self.P
        return self.x[:2].copy()

    def d2(self, z):
        """Distância de Mahalanobis ao quadrado entre z e a predição."""
        y = np.asarray(z, float) - self.M @ self.x
        return float(y @ np.linalg.inv(self.S()) @ y)

    def regiao(self, gate):
        """Elipse de busca: (semi-eixo a, semi-eixo b, ângulo em graus). Para desenhar."""
        w, V = np.linalg.eigh(self.S())
        a, b = np.sqrt(gate * w[1]), np.sqrt(gate * w[0])
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        return float(a), float(b), float(ang)


class Rastreador:
    """Cuida do ciclo: predição -> gating -> correção, e das falhas de detecção.

    estado de saída:
      'medido'    medição aceita e usada na correção
      'no_escuro' sem medição: só predição
      'rejeitado' havia medição, mas fora da região de busca (espúria): só predição
      'perdido'   passou de max_sem_medicao quadros no escuro: trilha reiniciada
    """

    def __init__(self, sd=3.0, sm=8.0, max_sem_medicao=30, gate=9.21, dt=1.0):
        # gate = 9.21 é o limite qui-quadrado (2 graus de liberdade, 99%)
        self.p = dict(dt=dt, sd=sd, sm=sm)
        self.max_sem, self.gate = max_sem_medicao, gate
        self.kf, self.falhas = None, 0

    def step(self, z):
        if self.kf is None:                      # sem trilha: espera 1ª detecção
            if z is None:
                return None
            self.kf, self.falhas = KalmanCV(z, **self.p), 0
            return dict(pred=tuple(z), est=tuple(z), estado="medido", regiao=None)

        pred = self.kf.predict()
        regiao = self.kf.regiao(self.gate)       # região de busca deste quadro
        if z is not None and self.kf.d2(z) <= self.gate:
            est, estado, self.falhas = self.kf.update(z), "medido", 0
        else:
            est = self.kf.x[:2].copy()           # só predição
            estado = "no_escuro" if z is None else "rejeitado"
            self.falhas += 1
            if self.falhas > self.max_sem:       # desistiu: reinicia na próxima detecção
                self.kf = None
                return dict(pred=tuple(pred), est=None, estado="perdido", regiao=regiao)
        return dict(pred=tuple(pred), est=tuple(est), estado=estado, regiao=regiao)
