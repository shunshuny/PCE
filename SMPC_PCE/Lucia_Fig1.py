import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pce_controller import PCEController
import cvxpy as cp

# システム定義とGPC行列の構築
def A_delta(delta):
    return np.array([
    [0.239,     0.0,     0.178,     0.0,     0.0],
    [-0.372*delta, 1.0,  0.250,     0.0,     0.0],
    [-0.990,    0.0,     0.139,     0.0,     0.0],
    [-48.9*delta, 64.1,   2.40,     1.0,     0.0],
    [0.0,       0.0,       0.0,     0.0,     0.0]
])

def B_delta(delta):
    return np.array([
    [-1.23*delta],
    [-1.44*delta],
    [-4.48*delta],
    [-1.80*delta],
    [1.0*delta]
])
    
# パラメータ定義
p_order = 4
Q = np.diag([1014.7, 3.2407, 5674.8, 0.3695, 471.75])
R = np.array([[4716.5]])
S = Q
s = 0
beta = 0.05
kappa = cp.sqrt((1-beta)/beta)

# インスタンス生成（関数の参照を渡す！）
controller = PCEController(A_delta, B_delta, Q, R, p_order)
# gPC行列の計算
Agpc = controller.Agpc
Bgpc = controller.Bgpc
Qx_bar = controller.Qx_bar
Ru_bar = controller.Ru_bar
p_terms = controller.p_terms
n = controller.n
m = controller.m

# 状態方程式
def next_x(x,u):
    return Agpc @ x + Bgpc @ u

# Constraints
def make_big_vec(v):
    v = v.flatten()  # 2次元の場合も1次元にする
    dimension = len(v)
    v0 = np.zeros([p_terms*dimension])
    for i in range(dimension):
        v0[i*(p_terms)] = v[i]
    return v0
G = np.array([[0, 1, 0, 0, 1]])  # Pitch angle constraint
G_big = make_big_vec(G)
g = np.array([[0, 0.349, 0, 0, 0.262]])   # Upper bound (20 degrees in radians)
g_big = make_big_vec(g)
H = np.array([[1], [-1]]) #入力への制約
H_big = np.array([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]])
h = np.array([0.262])
h_big = make_big_vec(h)

# Initial state
x0 = np.zeros([p_terms*n]) # p_terms*nと同じ次元
initial_list = [1.0, 0.1, 1.0, -400.0, 0.1]
for i in range(p_terms):
    x0[i*(p_terms)] = initial_list[i]

# Prediction horizon
N = 10
nx = n*p_terms
nu = m*p_terms

# Optimization variables
U = cp.Variable((nu, N))
X = cp.Variable((nx, N + 1))

# Cost function
cost = 0
constraints = [X[:, 0] == x0.flatten()]

# Dynamics and cost
for k in range(N):
    cost += cp.quad_form(X[:, k], Qx_bar) + cp.quad_form(U[:, k], Ru_bar) # s = 0
    constraints += [X[:, k + 1] == Agpc @ X[:, k] + Bgpc @ U[:, k]]


# State and input constraints
b_1 = 0.349
b_2 = 0.262
for k in range(N):
    # Linearized chance constraint (Equation (14) in the paper)
    # x2 =[]; x5 = []
    # for i in range(p_terms):
    #     x2.append(X[p_terms + i:k])
    #     x5.append(X[4*p_terms + i:k])
    # x2 = np.array(x2)
    # x5 = np.array(x5)  
    # mean_x2 = x2[0]
    # mean_x5 = x5[0]
    # V2 = controller.var_x(x2)
    # V5 = controller.var_x(x5)
    # chance_constraint_1 = 4*((mean_x2 + b_1)*(b_1 - mean_x2) - V2)/((2*b_1)**2)
    # chance_constraint_2 = 4*((mean_x5 + b_2)*(b_2 - mean_x5) - V5)/((2*b_2)**2)
    constraints += [X[1*p_terms, k] <= 0.346]
    constraints += [X[4*p_terms, k] <= 0.262]
    # constraints += [-0.346 = X[1*p_terms, k]]
    # constraints += [-0.262 = X[4*p_terms, k]]
    constraints += [H_big[0] @ U[:, k] <= h_big]
    constraints += [H_big[1] @ U[:, k] <= h_big]


# Solve optimization
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()
print(problem.status)
# Results
print('Optimal Control Inputs:', U.value)
print('Optimal State Trajectory:', X.value)
