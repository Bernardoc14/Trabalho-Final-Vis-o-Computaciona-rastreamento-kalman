"""Valida o kalman.py contra cv2.KalmanFilter (os resultados devem coincidir)."""
import cv2
import numpy as np
from kalman import KalmanCV

rng = np.random.default_rng(0)
N, SD, SM, VEL0 = 150, 3.0, 8.0, 20.0
t = np.arange(N)
verdade = np.c_[320 + 200 * np.sin(0.04 * t), 240 + 100 * np.sin(0.09 * t)]
z = verdade + rng.normal(0, SM, verdade.shape)
z[60:85] = np.nan                                  # oclusão: sem medição

# --- implementação da equipe ---
meu = KalmanCV(z[0], sd=SD, sm=SM, vel0=VEL0)
res_meu = [z[0]]
for k in range(1, N):
    meu.predict()
    res_meu.append(meu.update(z[k]) if not np.isnan(z[k, 0]) else meu.x[:2].copy())

# --- cv2.KalmanFilter com os MESMOS parâmetros ---
kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
kf.transitionMatrix = meu.D.copy()
kf.measurementMatrix = meu.M.copy()
kf.processNoiseCov = meu.Sd.copy()
kf.measurementNoiseCov = meu.Sm.copy()
kf.statePost = np.array([[z[0, 0]], [z[0, 1]], [0], [0]])
kf.errorCovPost = np.diag([SM**2, SM**2, VEL0**2, VEL0**2])
res_cv = [z[0]]
for k in range(1, N):
    kf.predict()
    if not np.isnan(z[k, 0]):
        est = kf.correct(z[k].reshape(2, 1))
    else:
        est = kf.statePre
    res_cv.append(est[:2, 0].copy())

dif = np.abs(np.array(res_meu) - np.array(res_cv)).max()
print(f"Maior diferença entre as duas implementações: {dif:.2e} pixels")
print("OK: resultados coincidem." if dif < 1e-6 else "ATENÇÃO: resultados diferentes!")
