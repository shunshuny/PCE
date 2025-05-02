import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy import signal

# --- Step 1: System Definition ---
# Random variable Delta ~ Uniform(-1, 1)  # [-1, 1]の一様分布

# Nominal system matrices (only A depends on Delta)
A_nominal = np.array([
    [-0.6398,  0.9378, -0.0014],
    [-1.5679, -0.8791, -0.1137],
    [ 0.0,     0.0,   -20.2   ]
])

B = np.array([
    [0.0],
    [0.0],
    [20.2]
])

C = np.array([[0.0, 180/np.pi, 0.0]])

# A(Delta) with parameter uncertainty (10% uncertainty)
def A_delta(delta):
    A = A_nominal.copy()
    A[1, 0] *= (1 + 0.1 * delta)
    A[1, 1] *= (1 + 0.1 * delta)
    A[1, 2] *= (1 + 0.1 * delta)
    return A

# --- Step 2: Controller Definition (transfer function) ---
# u = (0.3122s + 0.5538) / (s^2 + 2.128s + 1.132) * q
num = [0.3122, 0.5538]
den = [1.0, 2.128, 1.132]

# State-space realization of controller
Ac, Bc, Cc, Dc = signal.tf2ss(num, den) # 伝達関数 →　状態方程式

# --- Step 3: Closed-loop System Construction ---
def Acl(delta):
    A = A_delta(delta)
    Acl = np.block([
        [A,         B @ Cc.reshape(1, -1)],
        [Bc @ C,    Ac]
    ])
    return Acl

# --- Step 4: gPC Approximation (10 terms) ---
# For uniform distribution, use Legendre polynomials
from numpy.polynomial.legendre import legval

p_order = 4  # 10th order => 11 terms (0 to 10)
p_terms = p_order + 1

# Compute Galerkin projection of Acl(Delta) onto Legendre basis
# Integration over Delta in [-1, 1]
def compute_gpc_A_matrix(N_quad=50): # Nは数値積分用の分割数
    from numpy.polynomial.legendre import leggauss
    xi, wi = leggauss(N_quad)  # Gauss-Legendre quadrature points(xi) and weights(wi) on [-1, 1]

    n = Acl(0.0).shape[0] # Aclの次数を取得
    Agpc = np.zeros((n*p_terms, n*p_terms)) # n*(p+1)次の巨大行列

    # Basis evaluation cache
    Phi = np.zeros((p_terms, len(xi)))
    for k in range(p_terms):
        coeffs = np.zeros(k+1)
        coeffs[-1] = 1.0
        Phi[k, :] = legval(xi, coeffs)
    # print(Phi)

    for i in range(p_terms):
        for j in range(p_terms):
            Aij = np.zeros((n, n))
            for l, (xil, wl) in enumerate(zip(xi, wi)):
                A_l = Acl(xil)
                Aij += wl * Phi[i, l] * Phi[j, l] * A_l # 式(9)の分子
            min_ij = min(i,j)
            norm = 2.0 / (2*min_ij + 1) # Legendre多項式の自己内積(分母)
            Aij = Aij / norm
            # Fill into Agpc
            Agpc[i*n:(i+1)*n, j*n:(j+1)*n] = Aij

    return Agpc

Agpc = compute_gpc_A_matrix()
# print(Agpc) # n*(p+1)次の巨大行列

# --- Step 5: Monte Carlo Sampling ---
N_samples = 1000

delta_samples = np.random.uniform(-1, 1, N_samples)
mc_eigenvalues = []
for delta in delta_samples:
    eigvals = la.eigvals(Acl(delta))
    mc_eigenvalues.append(eigvals)
mc_eigenvalues = np.array(mc_eigenvalues)

# --- Step 6: Eigenvalue Computation ---
gpc_eigenvalues = la.eigvals(Agpc)
# print(gpc_eigenvalues) # n*(p+1)個の固有値

# --- Step 7: Plotting ---
plt.figure(figsize=(8,6))

# Flatten Monte Carlo eigenvalues
plt.plot(mc_eigenvalues.real.flatten(), mc_eigenvalues.imag.flatten(), 'r.', alpha=0.5, markersize=3, label='Monte Carlo')

# gPC eigenvalues with unfilled circles
plt.plot(gpc_eigenvalues.real, gpc_eigenvalues.imag, 'ko', markersize=8, markerfacecolor='none', label='gPC')

plt.xlabel('Real')
plt.ylabel('Imag')
plt.grid(True)
plt.title('Closed loop eigenvalue variations (gPC vs MC)')
plt.legend()
plt.xlim([-2.0, 0.0])
plt.ylim([-2.5, 2.5])
plt.show()
