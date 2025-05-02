# PCEの計算
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import legendre
from scipy.integrate import solve_ivp, quad
from numpy.polynomial.legendre import legval, legvander, leggauss

# Nominal system matrices (only A depends on Delta)
A_nominal = np.array([-2.0])

def cal_coeffs(p_terms):
    # モデル関数: f(x) = x - 2
    def model(x):
        return x + A_nominal

    # サンプル点（[-1, 1] の一様分布を想定）
    N = 100
    x = np.random.uniform(-1, 1, N)
    y = model(x)

    # Legendre多項式の基底（次数p_termsまで）
    V = legvander(x, p_terms)  # Vandermonde matrix: 各点での多項式評価

    # 最小二乗で係数を推定
    coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)
    # print("PCE（Legendre基底）係数:", coeffs)
    return coeffs

def legendre_inner_product(i, j, k):
    # ルジャンドル多項式の生成
    P_i = legendre(i)
    P_j = legendre(j)
    P_k = legendre(k)

    # 積分対象の関数
    integrand = lambda x: P_i(x) * P_j(x) * P_k(x)

    # 区間 [-1, 1] で数値積分
    result, _ = quad(integrand, -1, 1)
    return result

# A(Delta) with parameter uncertainty (10% uncertainty)
def A_delta(delta):
    A = A_nominal.copy()
    A[0] +=  delta
    return A

p_order = 2  # 10th order => 11 terms (0 to 10)
p_terms = p_order + 1

def gpc_A_matrix(): # Aがスカラーの場合
    Agpc = np.zeros((p_terms, p_terms))
    Phi = np.zeros((p_terms, p_terms, p_terms))
    
    for k in range(p_terms):
        for i in range(p_terms):
            norm = 2/(2*i + 1)
            for j in range(i, p_terms):
                Phi[i,j,k] = legendre_inner_product(i, j, k) / norm
                # print(i,j,k, '=', Phi[i,j,k])
        Phi_diag = np.diag(Phi[:,:,k])
        Phi[:,:,k] = Phi[:,:,k] + Phi[:,:,k].T - np.diag(Phi_diag)
        # print(Phi[:,:,k])
    # print(Phi[:,:,0])
    coeffes = cal_coeffs(p_order)
    # print(coeffes)
    for i in range(p_terms):
        Agpc += coeffes[i] * Phi[:,:,i]
    return Agpc

Agpc = gpc_A_matrix()
print(Agpc)
    
#--------------------------------------------------------------------------------------------------------
def compute_gpc_A_matrix(N_quad=50): # Nは数値積分用の分割数
    xi, wi = leggauss(N_quad)  # Gauss-Legendre quadrature points(xi) and weights(wi) on [-1, 1]

    n = A_delta(0.0).shape[0] # Aの次数を取得
    Agpc = np.zeros((n*p_terms, n*p_terms)) # n*(p+1)次の巨大行列

    # Basis evaluation cache
    Phi = np.zeros((p_terms, len(xi)))
    for k in range(p_terms):  # Phi_kの積分点数値を格納していく
        coeffs = np.zeros(k+1) # Legendre多項式の係数ベクトルを準備
        coeffs[-1] = 1.0
        Phi[k, :] = legval(xi, coeffs)
    # print(Phi)

    for i in range(p_terms):
        for j in range(p_terms):
            Aij = np.zeros((n, n))
            for l, (xil, wl) in enumerate(zip(xi, wi)):
                A_l = A_delta(xil)
                Aij += wl * Phi[i, l] * Phi[j, l] * A_l # 式(9)の分子
            min_ij = min(i,j)
            norm = 2.0 / (2*min_ij + 1) # Legendre多項式の自己内積(分母)
            Aij = Aij / norm
            # Fill into Agpc
            Agpc[i*n:(i+1)*n, j*n:(j+1)*n] = Aij

    return Agpc
# Agpc = compute_gpc_A_matrix()
# gpc_eigenvalues = la.eigvals(Agpc)
# print(Agpc)
# print(gpc_eigenvalues) # n*(p+1)個の固有値

# 初期状態 x(0)
x0 = np.array([1.0, 0.0, 0.0]) # p_ermsと同じ次元

# 時間設定
t_span = (0, 10)  # 0〜10秒
t_eval = np.linspace(t_span[0], t_span[1], 500)  # 描画用の時刻点

def linear_ode(t, x, Agpc):
    return Agpc @ x

# 微分方程式を数値的に解く
sol = solve_ivp(linear_ode, t_span, x0, args=(Agpc,), t_eval=t_eval)

# 平均と分散のプロット
mean = sol.y[0]
var = np.zeros_like(sol.y[0])
for i in range(1, p_terms):
    norm = 2 / (2*i + 1)  # Legendreの内積
    var += sol.y[i]**2 * norm

# モンテカルロ法を使った計算
N_samples = 1000
delta_samples = np.random.uniform(-1, 1, N_samples)
x_mc_all = np.zeros((N_samples, len(t_eval)))
for i, delta in enumerate(delta_samples):
    Amc = A_delta(delta)
    def mc_ode(t, x): return Amc @ x
    x0_mc = np.array([1.0 + 0.0 * delta])
    sol_mc = solve_ivp(mc_ode, t_span, x0_mc, t_eval=t_eval)
    x_mc_all[i, :] = sol_mc.y[0]
    
# MC平均・分散
mean_mc = np.mean(x_mc_all, axis=0)
var_mc = np.var(x_mc_all, axis=0)


plt.plot(sol.t, mean, label="Mean", color="blue")
plt.fill_between(sol.t, mean - np.sqrt(var), mean + np.sqrt(var), alpha=0.3, color="blue", label="±1σ")

plt.plot(t_eval, mean_mc, label="MC Mean", color="red", linestyle="--")
plt.fill_between(t_eval, mean_mc - np.sqrt(var_mc), mean_mc + np.sqrt(var_mc),
                 alpha=0.3, color="red", label="MC ±1σ")

plt.xlabel("Time [s]")
plt.ylabel("State")
plt.grid(True)
plt.legend()
plt.title("PCE solution: mean and variance band")
plt.show()
