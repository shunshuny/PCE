# pce_controller.py
import numpy as np
import jax.numpy as jnp
from jax import jacfwd
import scipy.linalg as la
from scipy.special import legendre
from scipy.integrate import quad, solve_ivp
from numpy.polynomial.legendre import legvander
from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from numpy.linalg import norm, inv
from scipy.integrate import solve_ivp

class PCEController:
    def __init__(self, A_delta, B1_delta, B2_delta, Q1, Q2, R11, R12, R21, R22, p_order):
        # 各変数の定義
        self.A_delta = A_delta
        self.B1_delta = B1_delta
        self.B2_delta = B2_delta
        self.Q1 = Q1
        self.Q2 = Q2
        self.R11 = R11
        self.R12 = R12
        self.R21 = R21
        self.R22 = R22
        self.p_order = p_order
        self.p_terms = p_order + 1
        self.n = A_delta(0.0).shape[0]
        self.m = B1_delta(0.0).shape[1]
        self.W = np.eye(self.p_terms)
        self.Q1_bar = np.kron(self.Q1, self.W)
        self.Q2_bar = np.kron(self.Q2, self.W)
        self.R11_bar = np.kron(self.R11, self.W)
        self.R12_bar = np.kron(self.R12, self.W)
        self.R21_bar = np.kron(self.R21, self.W)
        self.R22_bar = np.kron(self.R22, self.W)
        self.Agpc = self.gpc_matrix(self.cal_coeffs_A, self.n, self.n)
        self.B1gpc = self.gpc_matrix(self.cal_coeffs_B1, self.n, self.m)
        self.B2gpc = self.gpc_matrix(self.cal_coeffs_B2, self.n, self.m)

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

    def cal_coeffs_B1(self):
        # BのgPC係数を計算
        N = 100
        x = np.random.uniform(-1, 1, N)
        coeffs = np.zeros((self.n, self.m, self.p_terms))
        for i in range(self.n):
            for j in range(self.m):
                y = np.array([self.B1_delta(x_k)[i, j] for x_k in x])
                V = legvander(x, self.p_order)
                coeffs[i, j], *_ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs
    
    def cal_coeffs_B2(self):
        # BのgPC係数を計算
        N = 100
        x = np.random.uniform(-1, 1, N)
        coeffs = np.zeros((self.n, self.m, self.p_terms))
        for i in range(self.n):
            for j in range(self.m):
                y = np.array([self.B2_delta(x_k)[i, j] for x_k in x])
                V = legvander(x, self.p_order)
                coeffs[i, j], *_ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs

    def legendre_inner_product(self, i, j, k):
        # ルジャンドル多項式の生成
        Pi, Pj, Pk = legendre(i), legendre(j), legendre(k)
        # 積分対象の関数
        integrand = lambda x: Pi(x) * Pj(x) * Pk(x)
        # 区間 [a, b] で数値積分
        result, _ = quad(integrand, -1, 1)
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
                        norm = 2 / (2 * i + 1)
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
                        norm = 2/(2*i + 1)
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
                        norm = 2/(2*i + 1)
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
        count = 0
        for _ in range(N_mc):
            delta = np.random.uniform(-1, 1)
            A = self.A_delta(delta)
            B1 = self.B1_delta(delta)
            B2 = self.B2_delta(delta)
            P1, P2 = self.solve_riccati_newton(A, B1, B2, self.Q1, self.Q2, self.R11, self.R12, self.R21, self.R22, verbose=False)
            K1_mc = - inv(self.R11) @ B1.T @ P1
            K2_mc = - inv(self.R22) @ B2.T @ P2
            Acl = A + B1 @ K1_mc + B2 @ K2_mc
            sol = solve_ivp(lambda t, x: Acl @ x, t_span, initial_list, t_eval=t_eval)
            eigs_mc = np.linalg.eigvals(Acl)
            mc_eigenvalues.append(eigs_mc)
            x_mc_all.append(sol.y)
            count += 1
            print(f"モンテカルロ法{count}回目")
        x_mc_all = np.stack(x_mc_all, axis=2)
        mean_mc = np.mean(x_mc_all, axis=2)
        var_mc = np.var(x_mc_all, axis=2)
        return mc_eigenvalues, mean_mc, var_mc

    def mean_x(self, x):
        x_mean = []
        for i in range(self.n):
            x_mean.append(x[(i-1)*self.p_terms])
        return x_mean
    
    def var_x(self, x):
        var_x = 0 
        for i in range(1, self.p_terms):
            norm = self.legendre_inner_product(i, i, 0)  # Legendreの内積
            var_x += (x[i]**2) * norm
        return var_x
    
    def solve_riccati_newton(self, A, B1, B2, Q1, Q2, R11, R12, R21, R22, tol=1e-10, max_iter=100, verbose=True):
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
                if verbose:
                    print(f"収束しました。反復回数: {i+1}")
                break
            if i == max_iter - 1:
                print(f"収束しませんでした。反復回数: {i+1}")
                print(f"ノルム: {np.linalg.norm(f)}")

        return P1, P2