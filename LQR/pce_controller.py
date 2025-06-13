# pce_controller.py
import numpy as np
import scipy.linalg as la
from scipy.special import legendre
from scipy.integrate import quad, solve_ivp
from numpy.polynomial.legendre import legvander

class PCEController:
    def __init__(self, A_delta, B_delta, Q, R, p_order):
        # 各変数の定義
        self.A_delta = A_delta
        self.B_delta = B_delta
        self.Q = Q
        self.R = R
        self.p_order = p_order
        self.p_terms = p_order + 1
        self.n = A_delta(0.0).shape[0]
        self.m = B_delta(0.0).shape[1]
        self.W = np.eye(self.p_terms)
        self.Qx_bar = np.kron(self.Q, self.W)
        self.Ru_bar = np.kron(self.R, self.W)
        self.Agpc = self.gpc_matrix(self.cal_coeffs_A, self.n, self.n)
        self.Bgpc = self.gpc_matrix(self.cal_coeffs_B, self.n, self.m)

    def cal_coeffs_A(self):
        # AのgPC係数を計算
        N = 100
        x = np.random.uniform(-1, 1, N)
        coeffs = np.zeros((self.n, self.n, self.p_terms))
        for i in range(self.n):
            for j in range(self.n):
                y = np.array([self.A_delta(x_k)[i, j] for x_k in x])
                V = legvander(x, self.p_order)
                coeffs[i, j], *_ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs

    def cal_coeffs_B(self):
        # BのgPC係数を計算
        N = 100
        x = np.random.uniform(-1, 1, N)
        coeffs = np.zeros((self.n, self.m, self.p_terms))
        for i in range(self.n):
            for j in range(self.m):
                y = np.array([self.B_delta(x_k)[i, j] for x_k in x])
                V = legvander(x, self.p_order)
                coeffs[i, j], *_ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs

    def legendre_inner_product(self, i, j, k):
        # ルジャンドル多項式の生成
        Pi, Pj, Pk = legendre(i), legendre(j), legendre(k)
        # 積分対象の関数
        integrand = lambda x: Pi(x) * Pj(x) * Pk(x) /2 # 一様分布の確率密度関数をかける
        # 区間 [-1, 1] で数値積分
        result, _ = quad(integrand, -1, 1) / 2
        return result

    def gpc_matrix(self, coeffs_func, rows, cols):
        coeffes_all = coeffs_func()
        blocks = []
        for s in range(rows):
            row = []
            for t in range(cols):
                Ggpc_st = np.zeros((self.p_terms, self.p_terms))
                Phi = np.zeros((self.p_terms, self.p_terms, self.p_terms))
                for k in range(self.p_terms):
                    for i in range(self.p_terms):
                        norm = 1 / (2 * i + 1)
                        for j in range(i, self.p_terms):
                            Phi[i, j, k] = self.legendre_inner_product(i, j, k) / norm
                    Phi_diag = np.diag(Phi[:, :, k])
                    Phi[:, :, k] = Phi[:, :, k] + Phi[:, :, k].T - np.diag(Phi_diag)
                coeffes = coeffes_all[s, t]
                for i in range(self.p_terms):
                    Ggpc_st += coeffes[i] * Phi[:, :, i]
                row.append(Ggpc_st)
            blocks.append(row)
        return np.block(blocks)
    
    def gpc_A_matrix(self): # Aが一般の行列
        blocks = []
        for s in range(self.n):
            row = []
            for t in range(self.n):
                Agpc_st = np.zeros((self.p_terms, self.p_terms))
                Phi = np.zeros((self.p_terms, self.p_terms, self.p_terms))
                
                for k in range(self.p_terms):
                    for i in range(self.p_terms):
                        norm = 1/(2*i + 1)
                        for j in range(i, self.p_terms):
                            Phi[i,j,k] = self.legendre_inner_product(i, j, k) / norm
                            # print(i,j,k, '=', Phi[i,j,k])
                    Phi_diag = np.diag(Phi[:,:,k])
                    Phi[:,:,k] = Phi[:,:,k] + Phi[:,:,k].T - np.diag(Phi_diag)
                    # print(Phi[:,:,k])
                # print(Phi[:,:,0])
                coeffes_all = self.cal_coeffs_A()
                coeffes = coeffes_all[s,t]
                # print(coeffes)
                for i in range(self.p_terms):
                    Agpc_st += coeffes[i] * Phi[:,:,i]
                row.append(Agpc_st)
            blocks.append(row)
        Agpc = np.block(blocks)
        return Agpc

    def gpc_B_matrix(self): # Aが一般の行列
        blocks = []
        for s in range(self.n):
            row = []
            for t in range(self.m):
                Bgpc_st = np.zeros((self.p_terms, self.p_terms))
                Phi = np.zeros((self.p_terms, self.p_terms, self.p_terms))
                
                for k in range(self.p_terms):
                    for i in range(self.p_terms):
                        norm = 1/(2*i + 1)
                        for j in range(i, self.p_terms):
                            Phi[i,j,k] = self.legendre_inner_product(i, j, k) / norm
                            # print(i,j,k, '=', Phi[i,j,k])
                    Phi_diag = np.diag(Phi[:,:,k])
                    Phi[:,:,k] = Phi[:,:,k] + Phi[:,:,k].T - np.diag(Phi_diag)
                    # print(Phi[:,:,k])
                # print(Phi[:,:,0])
                coeffes_all = self.cal_coeffs_B()
                coeffes = coeffes_all[s,t]
                # print(coeffes)
                for i in range(self.p_terms):
                    Bgpc_st += coeffes[i] * Phi[:,:,i]
                row.append(Bgpc_st)
            blocks.append(row)
        Bgpc = np.block(blocks)
        return Bgpc

    def lqr(self, A, B, Q, R):
        P = la.solve_continuous_are(A, B, Q, R)
        K = la.inv(R) @ B.T @ P
        eig = la.eigvals(A - B @ K)
        return P, K, eig

    def solve_dynamics(self, Acl_pc, x0, t_span, t_eval):
        def linear_ode(t, x):
            return Acl_pc @ x
        return solve_ivp(linear_ode, t_span, x0, t_eval=t_eval)

    def simulate_mc(self, N_mc, initial_list, t_span, t_eval):
        x_mc_all = []
        mc_eigenvalues = []
        for _ in range(N_mc):
            delta = np.random.uniform(-1, 1)
            A = self.A_delta(delta)
            B = self.B_delta(delta)
            P_mc, K_mc, eigs_mc = self.lqr(A, B, self.Q, self.R)
            Acl = A - B @ K_mc
            sol = solve_ivp(lambda t, x: Acl @ x, t_span, initial_list, t_eval=t_eval)
            mc_eigenvalues.append(eigs_mc)
            x_mc_all.append(sol.y)
        x_mc_all = np.stack(x_mc_all, axis=2)
        mean_mc = np.mean(x_mc_all, axis=2)
        var_mc = np.var(x_mc_all, axis=2)
        return mc_eigenvalues, mean_mc, var_mc
