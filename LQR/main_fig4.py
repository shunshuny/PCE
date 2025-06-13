import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pce_controller import PCEController

# システム定義とGPC行列の構築
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

# パラメータ定義
p_order = 4
Q = 2 * np.eye(A_delta(0.0).shape[0])
R = np.eye(B_delta(0.0).shape[1])

# インスタンス生成（関数の参照を渡す！）
controller = PCEController(A_delta, B_delta, Q, R, p_order)

# gPC行列の計算
Agpc = controller.Agpc
Bgpc = controller.Bgpc
Qx_bar = controller.Qx_bar
Ru_bar = controller.Ru_bar
p_terms = controller.p_terms
n = controller.n

P, K_big, eigs_gpc = controller.lqr(Agpc, Bgpc, Qx_bar, Ru_bar)
Acl_pc = Agpc - Bgpc @ K_big

# 微分方程式を解く
# 初期状態 x(0)
x0 = np.zeros([p_terms*n]) # p_terms*nと同じ次元
initial_list = [1.0, 1.0, 1.0, 1.0, 1.0]
for i in range(n):
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
    var_alpha += (sol.y[i+p_terms]**2) * norm

# モンテカルロで実装
# eigs_mc, mean_mc, var_mc = controller.simulate_mc(1000, initial_list, t_span, t_eval)

plt.plot(sol.t, mean_alpha, label="Mean", color="blue")
plt.fill_between(sol.t, mean_alpha - np.sqrt(var_alpha), mean_alpha + np.sqrt(var_alpha), alpha=0.3, color="blue", label="±1σ")

# plt.plot(t_eval, mean_mc[1], label="MC Mean", color="red", linestyle="--")
# plt.fill_between(t_eval, mean_mc[1] - np.sqrt(var_mc[1]), mean_mc[1] + np.sqrt(var_mc[1]),
#                  alpha=0.3, color="red", label="MC ±1σ")

plt.xlabel("Time [s]")
plt.ylabel("State")
plt.grid(True)
plt.legend()
plt.title("PCE solution: mean and variance band")
plt.show()