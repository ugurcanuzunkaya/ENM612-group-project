import numpy as np
import gurobipy as gp
from gurobipy import GRB


class MaxMinSeparability:
    def __init__(
        self,
        n_groups,
        n_hyperplanes_per_group,
        n_features,
        lambda_step=1e-4,
        beta=0.5,
        max_iter=100,
    ):
        """
        Parameters from the paper:
        n_groups (r): Number of groups (set I).
        n_hyperplanes_per_group (j): Number of hyperplanes in each group J_i.
        """
        self.r = n_groups
        self.j = n_hyperplanes_per_group
        self.n = n_features
        self.l = self.r * self.j  # Total number of hyperplanes
        self.total_vars = (self.n + 1) * self.l  # n weights + 1 bias for each plane

        # Parameters for Discrete Gradient Method (DGM)
        self.lambda_step = lambda_step
        self.beta = beta
        self.max_iter = max_iter

        # Create group structure (Map flat index to group index)
        # J_groups[i] = [plane_indices...]
        self.J_groups = []
        idx_counter = 0
        for _ in range(self.r):
            group_indices = list(
                range(idx_counter, idx_counter + self.j)
            )  # Hyperplane indices in each J_i group. E.g., [[0,1,2], [3,4,5]] => J_1 = [0,1,2], J_2 = [3,4,5]
            self.J_groups.append(group_indices)
            idx_counter += self.j

    def _unpack_variables(self, variables):
        """
        Extracts W and gamma (y) matrices from a flat vector (Optimization variables).
        Assumes variables format: [x1_1..x1_n, y1, x2_1..x2_n, y2, ...]
        """
        # In the paper: x^j in R^n and y_j in R^1.
        # Separate variables into X matrix of size (l, n) and Y vector of size (l,).
        # For convenience, we keep the variable vector in this format:
        # [w1_1, ..., w1_n, bias1, w2_1, ..., w2_n, bias2, ...]

        W = np.zeros((self.l, self.n))
        Biases = np.zeros(self.l)

        stride = self.n + 1
        for i in range(self.l):
            start = i * stride
            end_w = start + self.n
            W[i, :] = variables[start:end_w]
            Biases[i] = variables[end_w]

        return W, Biases

    def objective_function(self, variables, A, B):
        """
        Paper Equations (31) and (32).
        f(x,y) = f1(x,y) + f2(x,y)
        """
        W, Y = self._unpack_variables(variables)

        # --- Term 1: Error for Set A (Eq 31) ---
        # Objective: min_{j in Ji} (<x^j, a> - y_j) < -1 => (<x^j, a> - y_j + 1) < 0
        f1_sum = 0
        m = len(A)
        if m > 0:
            # Calculate scores for all a's and all planes: Matrix of shape (m, l)
            # scores_A[k, h] = <x^h, a^k> - y_h + 1
            scores_A = np.dot(A, W.T) - Y + 1

            # Max-Min logic: max_{i in I} min_{j in Ji}
            point_errors = []
            for k in range(m):
                group_mins = []
                for group_idxs in self.J_groups:
                    # Minimum of planes within this group
                    val = np.min(scores_A[k, group_idxs])
                    group_mins.append(val)
                # Maximum of groups
                max_of_mins = np.max(group_mins)
                # max(0, result)
                point_errors.append(max(0, max_of_mins))

            f1_sum = np.mean(point_errors)  # 1/m * sum

        # --- Term 2: Error for Set B (Eq 32) ---
        # Objective: min_{j in Ji} (<x^j, b> - y_j) > 1 => -<x^j, b> + y_j + 1 < 0
        f2_sum = 0
        p = len(B)
        if p > 0:
            # scores_B[k, h] = -<x^h, b^k> + y_h + 1
            scores_B = -np.dot(B, W.T) + Y + 1

            # Min-Max logic: min_{i in I} max_{j in Ji}
            point_errors_b = []
            for k in range(p):
                group_maxs = []
                for group_idxs in self.J_groups:
                    # Maximum of planes within this group
                    val = np.max(scores_B[k, group_idxs])
                    group_maxs.append(val)
                # Minimum of groups
                min_of_maxs = np.min(group_maxs)
                # max(0, result)
                point_errors_b.append(max(0, min_of_maxs))

            f2_sum = np.mean(point_errors_b)  # 1/p * sum

        return f1_sum + f2_sum

    def compute_discrete_gradient(self, x, A, B):
        """
        Definition 2 from the paper - Discrete Gradient Calculation.
        Returns a discrete gradient vector by choosing a random direction (g)
        or using deterministic coordinate axes at the given point x.
        """
        n_vars = self.total_vars
        g = np.random.randn(n_vars)  # Random direction
        g = g / np.linalg.norm(g)  # Normalize

        # In the paper, u is chosen from U(g, alpha). For simplicity, pick the largest component.
        u = np.argmax(np.abs(g))

        # Step sizes (Concepts of z(lambda) and e(beta) from the paper)
        # We apply a simple finite difference approach, but within DGM logic:
        # v_i = (f(x + lambda*g + delta*e_i) - f(x + lambda*g)) / delta

        grad = np.zeros(n_vars)
        base_point = x + self.lambda_step * g
        f_base = self.objective_function(base_point, A, B)

        # Perturbation for each coordinate
        delta = self.lambda_step * (self.beta**0)  # simplification of e_t(beta)

        for i in range(n_vars):
            if i == u:
                continue

            perturb_point = base_point.copy()
            # Coordinate change similar to H operator logic in the paper
            # Forward difference is common in practical DGM implementations:
            perturb_point[i] += delta
            f_new = self.objective_function(perturb_point, A, B)

            grad[i] = (f_new - f_base) / delta

        # Calculate residual for the u-th coordinate (Last formula in Def 2)
        # In practice, calculating all coordinates with finite difference gives similar results.
        # However, we can calculate u separately to stay true to the paper, or
        # use finite difference for stability:
        perturb_point_u = base_point.copy()
        perturb_point_u[u] += delta
        f_new_u = self.objective_function(perturb_point_u, A, B)
        grad[u] = (f_new_u - f_base) / delta

        return grad

    def solve_direction_finding_subproblem_gurobi(self, gradients):
        """
        Wolfe Method / Direction Finding Subproblem.
        Finds the point (nr_g) in the convex hull of the given set of gradients (bundle)
        that is closest to the origin.
        Descent direction is d = -nr_g.

        Minimize || sum(c_k * v_k) ||^2
        s.t. sum(c_k) = 1, c_k >= 0
        """
        k = len(gradients)
        if k == 0:
            return np.zeros(self.total_vars)

        try:
            # Gurobi Model
            model = gp.Model("DirectionFinding")
            model.setParam("OutputFlag", 0)  # Silent mode

            # Variables: c_k (weights)
            c = model.addVars(k, lb=0.0, vtype=GRB.CONTINUOUS, name="c")

            # Constraint: Sum equal to 1
            model.addConstr(gp.quicksum(c[i] for i in range(k)) == 1.0, "SumOne")

            # Objective Function: || sum(c_k * v_k) ||^2
            # Expanding: sum_i sum_j c_i * c_j * (v_i . v_j)

            # Calculate Gram matrix (Dot products)
            V = np.array(gradients)  # (k, total_vars)
            Gram = np.dot(V, V.T)  # (k, k)

            obj = gp.QuadExpr()
            for i in range(k):
                for j in range(k):
                    obj += c[i] * c[j] * Gram[i, j]

            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()

            # Get optimal weights
            optimal_c = np.array([c[i].x for i in range(k)])

            # Create min-norm vector
            nr_g = np.dot(optimal_c, V)

            # Negative of the direction
            return -nr_g

        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return -gradients[-1]  # Fallback: reverse of last gradient

    def fit(self, X, y):
        """
        Training loop.
        """
        # Separate data into A and B sets
        A = X[y == 0]  # Class 0
        B = X[y == 1]  # Class 1 (A and B are disjoint sets in the paper)

        # Starting point
        current_vars = np.random.randn(self.total_vars) * 0.1

        gradients_bundle = []
        bundle_limit = 10  # For memory

        print(f"Initial Loss: {self.objective_function(current_vars, A, B):.4f}")

        for iter_num in range(self.max_iter):
            # 1. Compute Discrete Gradient
            grad = self.compute_discrete_gradient(current_vars, A, B)
            gradients_bundle.append(grad)
            if len(gradients_bundle) > bundle_limit:
                gradients_bundle.pop(0)

            # 2. Find Best Descent Direction (Gurobi)
            direction = self.solve_direction_finding_subproblem_gurobi(gradients_bundle)

            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-5:
                print("Approached optimum (Gradient norm very small).")
                break

            # 3. Line Search (Simple step-back similar to Armijo Rule)
            step_size = 1.0
            current_loss = self.objective_function(current_vars, A, B)
            found_step = False

            for _ in range(10):  # Max 10 attempts
                new_vars = current_vars + step_size * direction
                new_loss = self.objective_function(new_vars, A, B)

                if new_loss < current_loss:
                    current_vars = new_vars
                    current_loss = new_loss
                    found_step = True
                    break
                else:
                    step_size *= 0.5  # Shrink step

            if not found_step:
                # Clear bundle, look for new direction
                gradients_bundle = []

            if iter_num % 10 == 0:
                print(f"Iter {iter_num}: Loss = {current_loss:.5f}")
                if current_loss < 1e-4:
                    print("Perfect separation achieved.")
                    break

        self.optimized_vars = current_vars
        return self

    def predict(self, X, vars_override=None):
        """
        Predict determining which class the points belong to.
        According to Max-Min separability definition:
        Set A (Class 0): max_i min_j (score) < 0
        Set B (Class 1): min_i max_j (score) > 0

        In our model, set A is pushed to the negative region, and set B to the positive region.
        Therefore, if the function value is < 0, we can say Class A, if > 0, Class B.
        However, it is necessary to know 'which' max-min structure is active.

        A simple rule:
        If score_A calculated for point is low, it is close to A.
        If score_B calculated for point is low, it is close to B.
        """
        if vars_override is not None:
            vars_to_use = vars_override
        else:
            vars_to_use = self.optimized_vars

        W, Y = self._unpack_variables(vars_to_use)

        # Max-Min fonksiyonunu (separation function) hesaplayalım
        # Phi(z) = max_{i in I} min_{j in Ji} (<x^j, z> - y_j)
        # Makale Remark 1: A için Phi(a) < 0, B için Phi(b) > 0 olmalı.

        predictions = []
        for x_vec in X:
            # Tek bir nokta için skorlar: (l,)
            scores = np.dot(W, x_vec) - Y

            group_mins = []
            for group_idxs in self.J_groups:
                val = np.min(scores[group_idxs])
                group_mins.append(val)
            phi_val = np.max(group_mins)

            # Eşik değer 0
            if phi_val < 0:
                predictions.append(0)  # Class A
            else:
                predictions.append(1)  # Class B

        return np.array(predictions)
