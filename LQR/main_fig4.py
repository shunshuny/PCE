import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pce_controller import PceLqrController

# システム定義とGPC行列の構築
p_order = 4
initial_state = [1.0, 1.0, 1.0, 1.0, 1.0]
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 500)

controller = PceLqrController(p_order)
Agpc = controller.gpc_A_matrix()
Bgpc = controller.gpc_B_matrix()
P, K_big, eigs_gpc = controller.lqr(Agpc, Bgpc)
Acl_pc = Agpc - Bgpc @ K_big

# 初期状態の拡張（PCE空間）
p_terms = p_order + 1
n = controller.n
x0 = np.zeros([p_terms * n])
for i in range(p_terms):
    x0[i * n] = initial_state[i]

# PCE ODE解く
def linear_ode(t, x):
    return Acl_pc @ x

sol = solve_ivp(linear_ode, t_span, x0, t_eval=t_eval)

# 平均と分散を計算
def compute_mean_variance(sol_y, p_terms):
    mean_alpha = sol_y[controller.n]  # alpha_1の部分（0-indexed）
    var_alpha = np.zeros_like(mean_alpha)
    for i in range(1, p_terms):
        norm = 2 / (2*i + 1)
        var_alpha += sol_y[i * controller.n + 1] * norm
    return mean_alpha, var_alpha

mean_alpha, var_alpha = compute_mean_variance(sol.y, p_terms)

# モンテカルロとの比較
_, mean_mc, var_mc = controller.simulate_mc(N_mc=1000, initial_list=initial_state, t_span=t_span, t_eval=t_eval)

# 結果描画
plt.plot(sol.t, mean_alpha, label="Mean (PCE)", color="blue")
plt.fill_between(sol.t, mean_alpha - np.sqrt(var_alpha), mean_alpha + np.sqrt(var_alpha),
                 alpha=0.3, color="blue", label="PCE ±1σ")

plt.plot(t_eval, mean_mc[1], label="MC Mean", color="red", linestyle="--")
plt.fill_between(t_eval, mean_mc[1] - np.sqrt(var_mc[1]), mean_mc[1] + np.sqrt(var_mc[1]),
                 alpha=0.3, color="red", label="MC ±1σ")

plt.xlabel("Time [s]")
plt.ylabel("State")
plt.grid(True)
plt.legend()
plt.title("PCE vs Monte Carlo: Mean and Variance Band")
plt.show()
