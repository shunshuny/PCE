import numpy as np
import jax.numpy as jnp
from jax import jacfwd
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from numpy.linalg import norm, inv
from scipy.integrate import solve_ivp

def solve_riccati_newton(A, B1, B2, Q1, Q2, R11, R12, R21, R22, tol=1e-10, max_iter=100):
    """
    ニュートン法で連続時間型リカッチ方程式を解く
    A, B, Q, R: numpy配列
    tol: 収束許容誤差
    max_iter: 最大反復回数
    """
    n = A.shape[0]
    P1 = P2 = np.zeros((n, n))  # 初期解

    R11_inv = inv(R11); R12_inv = inv(R12); R21_inv = inv(R21); R22_inv = inv(R22)  
    S1 = B1 @ R11_inv @ B1.T; S2 = B2 @ R22_inv @ B2.T
    G1 = B1 @ R11_inv @ R21 @ R11_inv @ B1.T; G2 = B2 @ R22_inv @ R12 @ R22_inv @ B2.T 

    for i in range(max_iter):
        # 関数 F(X) の定義
        F1 = A.T @ P1 + P1 @ A - P1 @ S1 @ P1 - P1 @ S2 @ P2 - P2 @ S2 @ P1 + P2 @ G2 @ P2 + Q1  # リカッチ代数方程式の残差
        F2 = A.T @ P2 + P2 @ A - P2 @ S2 @ P2 - P2 @ S1 @ P1 - P1 @ S1 @ P2 + P1 @ G1 @ P1 + Q2

        # 1. 残差関数をベクトル化（f = [vec(F1); vec(F2)]）
        f1 = F1.reshape(-1)   # or F1.flatten()
        f2 = F2.reshape(-1)
        f = np.concatenate([f1, f2])  # サイズ: 2n^2

        # 2. ヤコビアンの計算（自作 or JAX/autograd で）
        def compute_jacobian(P1, P2):
            n = P1.shape[0]

            def residuals(vecP):
                # ベクトルを行列に変換
                P1_ = vecP[:n*n].reshape((n, n))
                P2_ = vecP[n*n:].reshape((n, n))

                # リカッチ残差（対象性は一旦無視して構成）
                F1 = A.T @ P1_ + P1_ @ A - P1_ @ S1 @ P1_ - P1_ @ S2 @ P2_ - P2_ @ S2 @ P1_ + P2_ @ G2 @ P2_ + Q1
                F2 = A.T @ P2_ + P2_ @ A - P2_ @ S2 @ P2_ - P2_ @ S1 @ P1_ - P1_ @ S1 @ P2_ + P1_ @ G1 @ P1_ + Q2

                return jnp.concatenate([F1.reshape(-1), F2.reshape(-1)])

            # 入力をベクトル化
            vecP0 = jnp.concatenate([P1.reshape(-1), P2.reshape(-1)])

            # 自動微分でヤコビアン計算
            J_fn = jacfwd(residuals)
            J = J_fn(vecP0)
            return J
        J = compute_jacobian(P1, P2)  # サイズ: 2n^2 × 2n^2

        # 3. ニュートンステップの解を求める
        delta_p = np.linalg.solve(J, -f)  # サイズ: 2n^2

        # 4. 解の分割・行列への整形
        delta_P1 = delta_p[:n*n].reshape((n, n))
        delta_P2 = delta_p[n*n:].reshape((n, n))

        # 5. 更新 ＋ 対称性を保つために symmetrize
        P1 = 0.5 * (P1 + delta_P1 + (P1 + delta_P1).T)
        P2 = 0.5 * (P2 + delta_P2 + (P2 + delta_P2).T)

        # 6. 収束判定（残差のノルム）
        if np.linalg.norm(f) < tol:
            print(f"収束しました。反復回数: {i+1}")
            break

    return P1, P2

# システム行列の例
A = np.array([[-1., 1.], [0., -2.]])
B1 = np.array([[1.], [1.]])
B2 = np.array([[1.], [1.]])
Q1 = np.array([[1., 0.2], [0.2, 1.0]])
Q2 = np.array([[1., 0.2], [0.2, 1.0]])
R11 = np.array([[1.]])
R12 = np.array([[1.]])
R21 = np.array([[1.]])
R22 = np.array([[1.]])

P1, P2 = solve_riccati_newton(A, B1, B2, Q1, Q2, R11, R12, R21, R22)
print("解 P1:")
print(P1)
print("解 P2:")
print(P2)

# 3. フィードバックゲイン計算
K1 = inv(R11) @ B1.T @ P1
K2 = inv(R22) @ B2.T @ P2
A_cl = A - B1 @ K1 - B2 @ K2  # 閉ループ系
eigvals = np.linalg.eigvals(A_cl)
print("A_cl の固有値:", eigvals)

# 4. シミュレーション
def dynamics(t, x):
    return A_cl @ x

x0 = np.array([1.0, 0.0])  # 初期状態
t_span = (0, 10)
t_eval = np.linspace(*t_span, 300)

sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval)

# 5. ノルムの時間変化を計算
x_t = sol.y
x_norms = np.linalg.norm(x_t, axis=0)

# 6. グラフ描画
plt.rcParams['font.family'] = 'Meiryo'
plt.figure(figsize=(8, 4))
plt.plot(sol.t, x_norms)
plt.xlabel('Time [s]')
plt.ylabel('||x(t)||')
plt.title('状態ベクトル x(t) のノルムの時間変化')
plt.grid(True)
plt.tight_layout()
plt.show()