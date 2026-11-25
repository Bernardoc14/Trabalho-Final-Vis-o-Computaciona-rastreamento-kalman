"""Valida o kalman.py contra cv2.KalmanFilter (os resultados devem coincidir).

Valida os dois modelos: velocidade constante (CV, 4 estados) e aceleração constante (CA, 6 estados).
    python validar_kalman.py
"""
import cv2
import numpy as np
from kalman import KalmanCV, KalmanCA

rng = np.random.default_rng(0)
N, SD, SM, VEL0, ACC0, SA = 150, 3.0, 8.0, 20.0, 2.0, 0.3
t = np.arange(N)
verdade = np.c_[320 + 200 * np.sin(0.04 * t), 240 + 100 * np.sin(0.09 * t)]
z = verdade + rng.normal(0, SM, verdade.shape)
z[60:85] = np.nan                                  # oclusão: sem medição


def comparar(nome, meu, n_estado, P0):
    # --- implementação da equipe ---
    res_meu = [z[0]]
    for k in range(1, N):
        meu.predict()
        res_meu.append(meu.update(z[k]) if not np.isnan(z[k, 0]) else meu.x[:2].copy())

    # --- cv2.KalmanFilter com os MESMOS parâmetros ---
    kf = cv2.KalmanFilter(n_estado, 2, 0, cv2.CV_64F)
    kf.transitionMatrix = meu_D = meu.D.copy()
    kf.measurementMatrix = meu.M.copy()
    kf.processNoiseCov = meu.Sd.copy()
    kf.measurementNoiseCov = meu.Sm.copy()
    kf.statePost = np.r_[z[0], np.zeros(n_estado - 2)].reshape(-1, 1)
    kf.errorCovPost = P0
    res_cv = [z[0]]
    for k in range(1, N):
        kf.predict()
        if not np.isnan(z[k, 0]):
            est = kf.correct(z[k].reshape(2, 1))
        else:
            est = kf.statePre
        res_cv.append(est[:2, 0].copy())

    dif = np.abs(np.array(res_meu) - np.array(res_cv)).max()
    print(f"[{nome}] maior diferença entre as duas implementações: {dif:.2e} pixels")
    print("   OK: resultados coincidem." if dif < 1e-6 else "   ATENÇÃO: resultados diferentes!")
    return dif < 1e-6


ok_cv = comparar("CV", KalmanCV(z[0], sd=SD, sm=SM, vel0=VEL0), 4,
                 np.diag([SM**2, SM**2, VEL0**2, VEL0**2]))
ok_ca = comparar("CA", KalmanCA(z[0], sd=SD, sm=SM, vel0=VEL0, acc0=ACC0, sa=SA), 6,
                 np.diag([SM**2, SM**2, VEL0**2, VEL0**2, ACC0**2, ACC0**2]))
raise SystemExit(0 if (ok_cv and ok_ca) else 1)