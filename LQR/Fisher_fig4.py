import numpy as np
import cvxpy as cp
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.linalg as la
from scipy.special import legendre
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve_continuous_are, block_diag
from numpy.polynomial.legendre import legval, leggauss, legvander

def cal_coeffs_A(p_order):
    # サンプル点（[-1, 1] の一様分布を想定）
    N = 100
    x = np.random.uniform(-1, 1, N)
    coeffs = np.zeros((n,n,p_order + 1))
    
    for i in range(n):
        for j in range(n):
            # 各サンプルに対して A_delta を計算し、(i,j) 成分を抽出
            y = np.array([A_delta(x_k)[i, j] for x_k in x])  # y は長さ N のベクトル
            V = legvander(x, p_order)  # N×(p_order+1) の基底評価行列
            # 最小二乗法で (i,j) 成分の PCE 係数を推定
            coeffs[i, j], *_ = np.linalg.lstsq(V, y, rcond=None)
            # print("PCE（Legendre基底）係数:", coeffs[i,j])
    return coeffs

def cal_coeffs_B(p_order):
    # サンプル点（[-1, 1] の一様分布を想定）
    N = 100
    x = np.random.uniform(-1, 1, N)
    coeffs = np.zeros((n,m,p_order + 1))
    
    for i in range(n):
        for j in range(m):
            # 各サンプルに対して A_delta を計算し、(i,j) 成分を抽出
            y = np.array([B_delta(x_k)[i, j] for x_k in x])  # y は長さ N のベクトル
            V = legvander(x, p_order)  # N×(p_order+1) の基底評価行列
            # 最小二乗法で (i,j) 成分の PCE 係数を推定
            coeffs[i, j], *_ = np.linalg.lstsq(V, y, rcond=None)
            # print("PCE（Legendre基底）係数:", coeffs[i,j])
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

def gpc_A_matrix(): # Aが一般の行列
    blocks = []
    for s in range(n):
        row = []
        for t in range(n):
            Agpc_st = np.zeros((p_terms, p_terms))
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
            coeffes_all = cal_coeffs_A(p_order)
            coeffes = coeffes_all[s,t]
            # print(coeffes)
            for i in range(p_terms):
                Agpc_st += coeffes[i] * Phi[:,:,i]
            row.append(Agpc_st)
        blocks.append(row)
    Agpc = np.block(blocks)
    return Agpc

def gpc_B_matrix(): # Aが一般の行列
    blocks = []
    for s in range(n):
        row = []
        for t in range(m):
            Bgpc_st = np.zeros((p_terms, p_terms))
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
            coeffes_all = cal_coeffs_B(p_order)
            coeffes = coeffes_all[s,t]
            # print(coeffes)
            for i in range(p_terms):
                Bgpc_st += coeffes[i] * Phi[:,:,i]
            row.append(Bgpc_st)
        blocks.append(row)
    Bgpc = np.block(blocks)
    return Bgpc

def lqr(Agpc, Bgpc, Qx_bar, Ru_bar):
    '''
    最適レギュレータ計算
    '''
    P = la.solve_continuous_are(Agpc, Bgpc, Qx_bar, Ru_bar)
    K = la.inv(Ru_bar).dot(Bgpc.T).dot(P)
    eig = la.eigvals(Agpc - Bgpc.dot(K))
    return P, K, eig

# Monte Carloサンプル数
def simulate_mc(N_mc=1000, initial_list=None, t_span=None, t_eval=None):
    if t_eval is None:
        raise ValueError("t_evalを指定してください。")
    if t_eval is None:
        raise ValueError("t_spanを指定してください。")
    if initial_list is None:
        raise ValueError("initial_list（初期状態ベクトル）を指定してください。")
    n = len(initial_list)
    x_mc_all = []
    mc_eigenvalues = []
    for _ in range(N_mc):
        delta = np.random.uniform(-1, 1)
        A = A_delta(delta)
        B = B_delta(delta)
        P_mc, K_mc, eigs_mc = lqr(A, B, Q, R) 
        Acl = A - B @ K_mc
        x0_mc = initial_list
        sol = solve_ivp(lambda t, x: Acl @ x, t_span, initial_list, t_eval=t_eval)
        mc_eigenvalues.append(eigs_mc)
        x_mc_all.append(sol.y) # shape: (n, len(t_eval))
    x_mc_all = np.stack(x_mc_all, axis=2)  # shape: (n, len(t_eval), N_mc)
    mean_mc = np.mean(x_mc_all, axis=2)
    var_mc = np.var(x_mc_all, axis=2)
    return mc_eigenvalues, mean_mc, var_mc
#--------------------------------------------------------------------------------------------------------
def A_delta(delta):
    return np.array([
    [0.1658, -13.1013, -7.2748 * (1 + 0.2 * delta),    -32.1739,  0.2780],
    [0.0018, -0.1301,  0.9276 * (1 + 0.2 * delta),     0.0,     -0.0012],
    [0.0,    -0.6436, -0.4763,  0.0,      0.0],
    [0.0,     0.0,     1.0,     0.0,      0.0],
    [0.0,     0.0,     0.0,     0.0,     -1.0]
])

def B_delta(delta):
    return np.array([
    [0.0,    -0.0706],
    [0.0,    -0.0004],
    [0.0,    -0.0157],
    [0.0,     0.0],
    [64.94,   0.0]
])

p_order = 4  # 10th order => 11 terms (0 to 10)
p_terms = p_order + 1
n = A_delta(0.0).shape[0] # Aの次数を取得
m = B_delta(0.0).shape[1] # Bの列数を取得

Agpc = gpc_A_matrix()
Bgpc = gpc_B_matrix()

Q = 2*np.eye(n)
R = np.eye(m)
W = np.eye(p_terms) # 単位行列でイイのかは謎
Qx_bar = np.kron(Q, W)
Ru_bar = np.kron(R, W)
Ip = np.eye(p_terms)

P, K_big, eigs_gpc = lqr(Agpc, Bgpc, Qx_bar, Ru_bar)
Acl_pc = Agpc - Bgpc @ K_big
# print(eigs_gpc)

# 微分方程式を解く
# 初期状態 x(0)
x0 = np.zeros([p_terms*n]) # p_terms*nと同じ次元
initial_list = [1.0, 1.0, 1.0, 1.0, 1.0]
for i in range(p_terms):
    x0[i*(p_terms)] = initial_list[i]
# 時間設定
t_span = (0, 20)  # 0〜10秒
t_eval = np.linspace(t_span[0], t_span[1], 500)  # 描画用の時刻点

def linear_ode(t, x, Acl_pc):
    return Acl_pc @ x

# 微分方程式を数値的に解く
sol = solve_ivp(linear_ode, t_span, x0, args=(Acl_pc,), t_eval=t_eval)
# 平均と分散の計算
mean_alpha = sol.y[p_terms]
var_alpha  = np.zeros_like(sol.y[p_terms])
for i in range(1, p_terms):
    norm = 2 / (2*i + 1)  # Legendreの内積
    var_alpha += sol.y[i+p_terms] * norm

eigs_mc, mean_mc, var_mc = simulate_mc(initial_list=initial_list, t_span=t_span, t_eval=t_eval)

plt.plot(sol.t, mean_alpha, label="Mean", color="blue")
plt.fill_between(sol.t, mean_alpha - np.sqrt(var_alpha), mean_alpha + np.sqrt(var_alpha), alpha=0.3, color="blue", label="±1σ")

plt.plot(t_eval, mean_mc[1], label="MC Mean", color="red", linestyle="--")
plt.fill_between(t_eval, mean_mc[1] - np.sqrt(var_mc[1]), mean_mc[1] + np.sqrt(var_mc[1]),
                 alpha=0.3, color="red", label="MC ±1σ")

plt.xlabel("Time [s]")
plt.ylabel("State")
plt.grid(True)
plt.legend()
plt.title("PCE solution: mean and variance band")
plt.show()