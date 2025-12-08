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
        Makaledeki parametreler:
        n_groups (r): I kümesinin eleman sayısı.
        n_hyperplanes_per_group (j): Her J_i kümesindeki hiperdüzlem sayısı.
        """
        self.r = n_groups
        self.j = n_hyperplanes_per_group
        self.n = n_features
        self.l = self.r * self.j  # Toplam hiperdüzlem sayısı
        self.total_vars = (
            self.n + 1
        ) * self.l  # Her düzlem için n tane w, 1 tane bias (y)

        # DGM Parametreleri
        self.lambda_step = lambda_step
        self.beta = beta
        self.max_iter = max_iter

        # Grup yapısını oluştur (Flat indeksten grup indeksine harita)
        # J_groups[i] = [düzlem_indeksleri...]
        self.J_groups = []
        idx_counter = 0
        for _ in range(self.r):
            group_indices = list(range(idx_counter, idx_counter + self.j))
            self.J_groups.append(group_indices)
            idx_counter += self.j

    def _unpack_variables(self, variables):
        """
        Düz bir vektörden (Optimization variables) W ve gamma (y) matrislerini çıkarır.
        variables: [x1_1..x1_n, y1, x2_1..x2_n, y2, ...] formatında varsayıyoruz.
        """
        # Makalede x^j \in R^n ve y_j \in R^1.
        # Değişkenleri (l, n) boyutunda X matrisi ve (l,) boyutunda Y vektörü olarak ayıralım.
        # Kolaylık olsun diye değişken vektörünü şu formatta tutacağız:
        # [w1_1, ..., w1_n, bias1, w2_1, ..., bias2, ...]

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
        Makale Denklem (31) ve (32).
        f(x,y) = f1(x,y) + f2(x,y)
        """
        W, Y = self._unpack_variables(variables)

        # --- Term 1: A kümesi için hata (Eq 31) ---
        # Amaç: min_{j in Ji} (<x^j, a> - y_j) < -1 => (<x^j, a> - y_j + 1) < 0
        f1_sum = 0
        m = len(A)
        if m > 0:
            # Tüm a'lar ve tüm düzlemler için skorları hesapla: (m, l) matrisi
            # scores_A[k, h] = <x^h, a^k> - y_h + 1
            scores_A = np.dot(A, W.T) - Y + 1

            # Max-Min mantığı: max_{i in I} min_{j in Ji}
            point_errors = []
            for k in range(m):
                group_mins = []
                for group_idxs in self.J_groups:
                    # Bu grubun içindeki düzlemlerin minimumu
                    val = np.min(scores_A[k, group_idxs])
                    group_mins.append(val)
                # Grupların maksimumu
                max_of_mins = np.max(group_mins)
                # max(0, sonuç)
                point_errors.append(max(0, max_of_mins))

            f1_sum = np.mean(point_errors)  # 1/m * sum

        # --- Term 2: B kümesi için hata (Eq 32) ---
        # Amaç: min_{j in Ji} (<x^j, b> - y_j) > 1 => -<x^j, b> + y_j + 1 < 0
        f2_sum = 0
        p = len(B)
        if p > 0:
            # scores_B[k, h] = -<x^h, b^k> + y_h + 1
            scores_B = -np.dot(B, W.T) + Y + 1

            # Min-Max mantığı: min_{i in I} max_{j in Ji}
            point_errors_b = []
            for k in range(p):
                group_maxs = []
                for group_idxs in self.J_groups:
                    # Bu grubun içindeki düzlemlerin maksimumu
                    val = np.max(scores_B[k, group_idxs])
                    group_maxs.append(val)
                # Grupların minimumu
                min_of_maxs = np.min(group_maxs)
                # max(0, sonuç)
                point_errors_b.append(max(0, min_of_maxs))

            f2_sum = np.mean(point_errors_b)  # 1/p * sum

        return f1_sum + f2_sum

    def compute_discrete_gradient(self, x, A, B):
        """
        Makale Tanım 2 (Definition 2) - Discrete Gradient Hesabı.
        Bu fonksiyon, verilen x noktasında rastgele bir yön (g) seçerek
        veya deterministik koordinat eksenleri kullanarak bir ayrık gradyan vektörü döndürür.
        """
        n_vars = self.total_vars
        g = np.random.randn(n_vars)  # Rastgele bir yön
        g = g / np.linalg.norm(g)  # Normalize et

        # Makalede u \in U(g, alpha) seçiliyor. Basitleştirme için en büyük bileşeni alalım.
        u = np.argmax(np.abs(g))

        # Adım büyüklükleri (Makaledeki z(lambda) ve e(beta) kavramları)
        # Basit bir sonlu farklar yaklaşımı uyguluyoruz ama DGM mantığında:
        # v_i = (f(x + lambda*g + delta*e_i) - f(x + lambda*g)) / delta

        grad = np.zeros(n_vars)
        base_point = x + self.lambda_step * g
        f_base = self.objective_function(base_point, A, B)

        # Her koordinat için perturbasyon
        delta = self.lambda_step * (self.beta**0)  # e_t(beta) basitleştirmesi

        for i in range(n_vars):
            if i == u:
                continue

            perturb_point = base_point.copy()
            # Makaledeki H operatörü mantığına benzer şekilde koordinat değişimi
            # Pratik DGM implementasyonlarında forward difference yaygındır:
            perturb_point[i] += delta
            f_new = self.objective_function(perturb_point, A, B)

            grad[i] = (f_new - f_base) / delta

        # u. koordinat için bakiye hesabı (Definition 2'deki son formül)
        # Genellikle pratikte tüm koordinatları finite difference ile hesaplamak da benzer sonuç verir
        # Ancak makaleye sadık kalmak için u'yu ayrı hesaplayabiliriz veya
        # kodun kararlılığı için onu da finite difference ile hesaplayalım:
        perturb_point_u = base_point.copy()
        perturb_point_u[u] += delta
        f_new_u = self.objective_function(perturb_point_u, A, B)
        grad[u] = (f_new_u - f_base) / delta

        return grad

    def solve_direction_finding_subproblem_gurobi(self, gradients):
        """
        Wolfe Method / Direction Finding Subproblem.
        Verilen gradyanlar kümesinin (bundle) konveks zarfında (convex hull)
        orijine en yakın noktayı (nr_g) bulur.
        İniş yönü d = -nr_g olur.

        Minimize || sum(c_k * v_k) ||^2
        s.t. sum(c_k) = 1, c_k >= 0
        """
        k = len(gradients)
        if k == 0:
            return np.zeros(self.total_vars)

        try:
            # Gurobi Model
            model = gp.Model("DirectionFinding")
            model.setParam("OutputFlag", 0)  # Sessiz mod

            # Değişkenler: c_k (ağırlıklar)
            c = model.addVars(k, lb=0.0, vtype=GRB.CONTINUOUS, name="c")

            # Kısıt: Toplamları 1 olmalı
            model.addConstr(gp.quicksum(c[i] for i in range(k)) == 1.0, "SumOne")

            # Amaç Fonksiyonu: || sum(c_k * v_k) ||^2
            # Bu ifadeyi açarsak: sum_i sum_j c_i * c_j * (v_i . v_j)

            # Gram matrisini (Dot products) hesapla
            V = np.array(gradients)  # (k, total_vars)
            Gram = np.dot(V, V.T)  # (k, k)

            obj = gp.QuadExpr()
            for i in range(k):
                for j in range(k):
                    obj += c[i] * c[j] * Gram[i, j]

            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()

            # En iyi ağırlıkları al
            optimal_c = np.array([c[i].x for i in range(k)])

            # Min-norm vektörünü oluştur
            nr_g = np.dot(optimal_c, V)

            # İniş yönü negatifi
            return -nr_g

        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return -gradients[-1]  # Fallback: son gradyanın tersi

    def fit(self, X, y):
        """
        Eğitim döngüsü.
        """
        # Veriyi A ve B kümelerine ayır
        A = X[y == 0]  # Class 0
        B = X[y == 1]  # Class 1 (Makalede A ve B disjoint kümeler)

        # Başlangıç noktası
        current_vars = np.random.randn(self.total_vars) * 0.1

        gradients_bundle = []
        bundle_limit = 10  # Hafıza için

        print(f"Başlangıç Loss: {self.objective_function(current_vars, A, B):.4f}")

        for iter_num in range(self.max_iter):
            # 1. Ayrık Gradyan Hesapla
            grad = self.compute_discrete_gradient(current_vars, A, B)
            gradients_bundle.append(grad)
            if len(gradients_bundle) > bundle_limit:
                gradients_bundle.pop(0)

            # 2. En İyi İniş Yönünü Bul (Gurobi)
            direction = self.solve_direction_finding_subproblem_gurobi(gradients_bundle)

            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-5:
                print("Optimuma yaklaşıldı (Gradient normu çok küçük).")
                break

            # 3. Line Search (Armijo Rule benzeri basit bir geri çekilme)
            step_size = 1.0
            current_loss = self.objective_function(current_vars, A, B)
            found_step = False

            for _ in range(10):  # Max 10 deneme
                new_vars = current_vars + step_size * direction
                new_loss = self.objective_function(new_vars, A, B)

                if new_loss < current_loss:
                    current_vars = new_vars
                    current_loss = new_loss
                    found_step = True
                    break
                else:
                    step_size *= 0.5  # Adımı küçült

            if not found_step:
                # Bundle'ı temizle, yeni bir yöne bak
                gradients_bundle = []

            if iter_num % 10 == 0:
                print(f"Iter {iter_num}: Loss = {current_loss:.5f}")
                if current_loss < 1e-4:
                    print("Mükemmel ayrışma sağlandı.")
                    break

        self.optimized_vars = current_vars
        return self

    def predict(self, X, vars_override=None):
        """
        Noktaların hangi sınıfa ait olduğunu tahmin et.
        Max-Min separability tanımına göre:
        A kümesi (Class 0): max_i min_j (score) < 0
        B kümesi (Class 1): min_i max_j (score) > 0 (gibi bir ayrım, ama burada skora bakacağız)

        Bizim modelde A kümesi negatif bölgeye, B kümesi pozitif bölgeye itildi.
        Dolayısıyla fonksiyonun değeri < 0 ise Class A, > 0 ise Class B diyebiliriz.
        Ancak burada 'hangi' max-min yapısının aktif olduğunu bilmek gerekir.

        Basit bir kural:
        Nokta a için hesaplanan score_A değeri düşükse A'ya yakındır.
        Nokta b için hesaplanan score_B değeri düşükse B'ye yakındır.
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
