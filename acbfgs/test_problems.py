"""
acbfgs - Adaptive Cautious BFGS Methods
========================================
Test problem library: classical CUTEst functions, modern nonconvex problems,
and structured stress tests, all implemented directly in Python.
"""
import numpy as np
from numpy.linalg import norm

# ============================================================================
# Part 1: Classical CUTEst Unconstrained Test Functions
# ============================================================================

class TestFunction:
    """Base class for test functions."""
    def __init__(self, name, n, x0=None, fstar=None):
        self.name = name
        self.n = n
        self.x0 = x0 if x0 is not None else np.ones(n)
        self.fstar = fstar  # known optimal value

    def eval(self, x):
        """Evaluate f(x)."""
        raise NotImplementedError

    def grad(self, x):
        """Evaluate gradient."""
        raise NotImplementedError

    def __call__(self, x):
        return self.eval(x)


# ---------- Extended Functions ----------

class ExtendedRosenbrock(TestFunction):
    """f(x) = sum_{i=1}^{n/2} [100(x_{2i-1}^2 - x_{2i})^2 + (x_{2i-1} - 1)^2]"""
    def __init__(self, n=100):
        super().__init__("Extended Rosenbrock", n, fstar=0.0)
        x0 = np.zeros(n)
        for i in range(0, n, 2):
            x0[i] = -1.2
            x0[i+1] = 1.0
        self.x0 = x0

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 2):
            f += 100.0 * (x[i]**2 - x[i+1])**2 + (x[i] - 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 2):
            g[i] = 400.0 * x[i] * (x[i]**2 - x[i+1]) + 2.0 * (x[i] - 1.0)
            g[i+1] = -200.0 * (x[i]**2 - x[i+1])
        return g


class ExtendedPowell(TestFunction):
    """f(x) = sum_{i=1}^{n/4} [(x_{4i-3}+10x_{4i-2})^2 + 5(x_{4i-1}-x_{4i})^2
           + (x_{4i-2}-2x_{4i-1})^4 + 10(x_{4i-3}-x_{4i})^4]"""
    def __init__(self, n=100):
        super().__init__("Extended Powell", n, fstar=0.0)
        x0 = np.zeros(n)
        for i in range(0, n, 4):
            x0[i] = 3.0
            x0[i+1] = -1.0
            x0[i+2] = 0.0
            x0[i+3] = 1.0
        self.x0 = x0

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 4):
            t1 = x[i] + 10.0 * x[i+1]
            t2 = x[i+2] - x[i+3]
            t3 = x[i+1] - 2.0 * x[i+2]
            t4 = x[i] - x[i+3]
            f += t1**2 + 5.0 * t2**2 + t3**4 + 10.0 * t4**4
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 4):
            t1 = x[i] + 10.0 * x[i+1]
            t2 = x[i+2] - x[i+3]
            t3 = x[i+1] - 2.0 * x[i+2]
            t4 = x[i] - x[i+3]
            g[i]     = 2.0 * t1 + 40.0 * t4**3
            g[i+1]   = 20.0 * t1 + 4.0 * t3**3
            g[i+2]   = 10.0 * t2 - 8.0 * t3**3
            g[i+3]   = -10.0 * t2 - 40.0 * t4**3
        return g


class ExtendedBeale(TestFunction):
    """Beale function extended: f(x) = sum_i (1.5 - x_{2i-1}(1-x_{2i}))^2
       + (2.25 - x_{2i-1}(1-x_{2i}^2))^2 + (2.625 - x_{2i-1}(1-x_{2i}^3))^2"""
    def __init__(self, n=100):
        super().__init__("Extended Beale", n, fstar=0.0)
        x0 = np.ones(n)
        self.x0 = x0

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 2):
            x1, x2 = x[i], x[i+1]
            f += (1.5 - x1*(1-x2))**2 + (2.25 - x1*(1-x2**2))**2 + (2.625 - x1*(1-x2**3))**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 2):
            x1, x2 = x[i], x[i+1]
            a1, a2, a3 = (1-x2), (1-x2**2), (1-x2**3)
            t1, t2, t3 = (1.5-x1*a1), (2.25-x1*a2), (2.625-x1*a3)
            g[i] = -2.0*(t1*a1 + t2*a2 + t3*a3)
            g[i+1] = 2.0*t1*x1 + 2.0*t2*2*x1*x2 + 2.0*t3*3*x1*x2**2
        return g


# ---------- Trigonometric / Periodic ----------

class Trigonometric(TestFunction):
    """f(x) = sum_i [n - sum_j cos(x_j) + i*(1-cos(x_i)) - sin(x_i)]^2"""
    def __init__(self, n=100):
        super().__init__("Trigonometric", n, fstar=0.0)
        self.x0 = np.ones(n) / n

    def eval(self, x):
        s = np.sum(np.cos(x))
        f = 0.0
        for i in range(self.n):
            f += (self.n - s + (i+1)*(1-np.cos(x[i])) - np.sin(x[i]))**2
        return f

    def grad(self, x):
        s = np.sum(np.cos(x))
        g = np.zeros(self.n)
        for i in range(self.n):
            ti = self.n - s + (i+1)*(1-np.cos(x[i])) - np.sin(x[i])
            g[i] = 2.0 * ti * ((i+1)*np.sin(x[i]) - np.cos(x[i]))
        for j in range(self.n):
            gj_extra = 0.0
            for i in range(self.n):
                ti = self.n - s + (i+1)*(1-np.cos(x[i])) - np.sin(x[i])
                gj_extra += ti
            g[j] += 2.0 * gj_extra * np.sin(x[j])
        return g


class FreudensteinRoth(TestFunction):
    """f(x) = (-13 + x1 + ((5-x2)*x2 - 2)*x2)^2
            + (-29 + x1 + ((x2+1)*x2 - 14)*x2)^2"""
    def __init__(self):
        super().__init__("Freudenstein & Roth", 2, np.array([0.5, -2.0]), fstar=0.0)

    def eval(self, x):
        f1 = -13.0 + x[0] + ((5.0-x[1])*x[1] - 2.0)*x[1]
        f2 = -29.0 + x[0] + ((x[1]+1.0)*x[1] - 14.0)*x[1]
        return f1**2 + f2**2

    def grad(self, x):
        f1 = -13.0 + x[0] + ((5.0-x[1])*x[1] - 2.0)*x[1]
        f2 = -29.0 + x[0] + ((x[1]+1.0)*x[1] - 14.0)*x[1]
        g1 = 2.0 * (f1 + f2)
        d1 = 10.0*x[1] - 3.0*x[1]**2 - 2.0
        d2 = 3.0*x[1]**2 + 2.0*x[1] - 14.0
        g2 = 2.0 * (f1*d1 + f2*d2)
        return np.array([g1, g2])


class BrownAlmostLinear(TestFunction):
    """f(x) = sum_{i=1}^{n-1} (x_i + sum_{j=1}^n x_j - (n+1))^2 + (prod_{j=1}^n x_j - 1)^2"""
    def __init__(self, n=100):
        super().__init__("Brown Almost-Linear", n, fstar=0.0)
        self.x0 = 0.5 * np.ones(n)

    def eval(self, x):
        s = np.sum(x)
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] + s - (self.n + 1))**2
        f += (np.prod(x) - 1)**2
        return f

    def grad(self, x):
        s = np.sum(x)
        prod = np.prod(x)
        g = np.zeros(self.n)
        for i in range(self.n):
            g[i] = 0.0
            for j in range(self.n - 1):
                g[i] += 2.0 * (x[j] + s - (self.n + 1)) * (1.0 if i == j else 0.0 + 1.0)
            g[i] += 2.0 * (prod - 1.0) * (prod / x[i] if x[i] != 0 else 0.0)
        return g


class Cosine(TestFunction):
    """f(x) = sum_{i=1}^{n-1} cos(-0.5*x_{i+1} + x_i^2)"""
    def __init__(self, n=100):
        super().__init__("Cosine", n, fstar=-99.0)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += np.cos(-0.5*x[i+1] + x[i]**2)
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            s = np.sin(-0.5*x[i+1] + x[i]**2)
            g[i] += -2.0 * x[i] * s
            g[i+1] += 0.5 * s
        return g


# ---------- Ill-conditioned ----------

class PenaltyI(TestFunction):
    """f(x) = (1e-5)*sum_{i=1}^n (x_i - 1)^2 + (sum_{i=1}^n x_i^2 - 0.25)^2"""
    def __init__(self, n=100):
        super().__init__("Penalty I", n, fstar=None)
        self.x0 = np.array([float(i+1) for i in range(n)])

    def eval(self, x):
        s1 = np.sum((x - 1.0)**2)
        s2 = np.sum(x**2) - 0.25
        return 1e-5 * s1 + s2**2

    def grad(self, x):
        s2 = np.sum(x**2) - 0.25
        return 2e-5 * (x - 1.0) + 4.0 * s2 * x


class PenaltyII(TestFunction):
    """f(x) = sum_{i=1}^{n-1} (x_i - 1)^2 + (sum_i x_i^2 - 0.25)^2"""
    def __init__(self, n=100):
        super().__init__("Penalty II", n, fstar=None)
        self.x0 = 0.5 * np.ones(n)

    def eval(self, x):
        s1 = np.sum((x[:-1] - 1.0)**2)
        s2 = np.sum(x**2) - 0.25
        return s1 + s2**2

    def grad(self, x):
        s2 = np.sum(x**2) - 0.25
        g = 2.0 * (x - 1.0)
        g[-1] = 0.0
        return g + 4.0 * s2 * x


class ARWHEAD(TestFunction):
    """f(x) = sum_{i=1}^{n-1} (-4*x_i + 3) + sum_{i=1}^{n-1} (x_i^2 + x_n^2)^2"""
    def __init__(self, n=100):
        super().__init__("ARWHEAD", n, fstar=None)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += -4.0 * x[i] + 3.0 + (x[i]**2 + x[-1]**2)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            t = x[i]**2 + x[-1]**2
            g[i] += -4.0 + 4.0 * x[i] * t
            g[-1] += 4.0 * x[-1] * t
        return g


class NONDIA(TestFunction):
    """f(x) = (1-x_1)^2 + sum_{i=2}^n 100*(x_1 - x_i^2)^2"""
    def __init__(self, n=100):
        super().__init__("NONDIA", n, fstar=0.0)
        self.x0 = -np.ones(n)

    def eval(self, x):
        f = (1.0 - x[0])**2
        for i in range(1, self.n):
            f += 100.0 * (x[0] - x[i]**2)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        g[0] = -2.0 * (1.0 - x[0])
        for i in range(1, self.n):
            g[0] += 200.0 * (x[0] - x[i]**2)
            g[i] = -400.0 * x[i] * (x[0] - x[i]**2)
        return g


# ---------- Sparse / Special ----------

class BroydenTridiagonal(TestFunction):
    """f(x) = sum_{i=1}^{n-1} ((3-2*x_i)*x_i - x_{i-1} - 2*x_{i+1} + 1)^2, x_0=x_{n+1}=0"""
    def __init__(self, n=100):
        super().__init__("Broyden Tridiagonal", n, fstar=0.0)
        self.x0 = -np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            xim1 = 0.0 if i == 0 else x[i-1]
            xip1 = x[i+1]
            t = (3.0 - 2.0*x[i])*x[i] - xim1 - 2.0*xip1 + 1.0
            f += t**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            xim1 = 0.0 if i == 0 else x[i-1]
            xip1 = x[i+1]
            t = (3.0 - 2.0*x[i])*x[i] - xim1 - 2.0*xip1 + 1.0
            g[i] += 2.0 * t * (3.0 - 4.0*x[i])
            if i > 0:
                g[i-1] += -2.0 * t
            g[i+1] += -4.0 * t
        return g


class BDQRTIC(TestFunction):
    """f(x) = sum_{i=1}^{n-4} (-4*x_i + 3) + sum_{i=1}^{n-4} (x_i^2 + 2*x_{i+1}^2
              + 3*x_{i+2}^2 + 4*x_{i+3}^2 + 5*x_n^2)^2"""
    def __init__(self, n=100):
        super().__init__("BDQRTIC", n, fstar=None)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 4):
            f += -4.0*x[i] + 3.0
            t = x[i]**2 + 2*x[i+1]**2 + 3*x[i+2]**2 + 4*x[i+3]**2 + 5*x[-1]**2
            f += t**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 4):
            g[i] += -4.0
            t = x[i]**2 + 2*x[i+1]**2 + 3*x[i+2]**2 + 4*x[i+3]**2 + 5*x[-1]**2
            g[i]   += 4.0 * x[i] * t
            g[i+1] += 8.0 * x[i+1] * t
            g[i+2] += 12.0 * x[i+2] * t
            g[i+3] += 16.0 * x[i+3] * t
            g[-1]  += 20.0 * x[-1] * t
        return g


# ---------- Miscellaneous ----------

class HelicalValley(TestFunction):
    """Fletcher's helical valley, n=3"""
    def __init__(self):
        super().__init__("Helical Valley", 3, np.array([-1.0, 0.0, 0.0]), fstar=0.0)

    def eval(self, x):
        theta = 0.5 * np.arctan2(x[1], x[0]) / np.pi
        if x[0] < 0:
            theta += 0.5 / np.pi
        t = x[0]**2 + x[1]**2
        return 100.0 * ((x[2] - 10.0*theta)**2 + (np.sqrt(t) - 1.0)**2) + x[2]**2

    def grad(self, x):
        t = x[0]**2 + x[1]**2
        theta = 0.5 * np.arctan2(x[1], x[0]) / np.pi
        if x[0] < 0:
            theta += 0.5 / np.pi
        dtheta_dx1 = -0.5 * x[1] / (2*np.pi*t)
        dtheta_dx2 = 0.5 * x[0] / (2*np.pi*t)
        f1 = x[2] - 10.0*theta
        f2 = np.sqrt(t) - 1.0
        g1 = 200.0 * (f1*(-10.0*dtheta_dx1) + f2*x[0]/np.sqrt(t))
        g2 = 200.0 * (f1*(-10.0*dtheta_dx2) + f2*x[1]/np.sqrt(t))
        g3 = 200.0 * f1 + 2.0*x[2]
        return np.array([g1, g2, g3])


class Chebyquad(TestFunction):
    """Chebyquad function"""
    def __init__(self, n=8):
        super().__init__("Chebyquad", n, fstar=0.0)
        x0 = np.array([float(i+1)/(n+1) for i in range(n)])
        self.x0 = x0
        self._y = np.zeros(n)
        for i in range(n):
            self._y[i] = float(i+1) / (n+1)

    def eval(self, x):
        f = 0.0
        for i in range(self.n):
            s = 0.0
            for j in range(self.n):
                s += self._T(i, 2.0*x[j]-1.0)
            f += (s)**2
        return f

    def _T(self, idx, z):
        if idx == 0:
            return 1.0
        elif idx == 1:
            return z
        else:
            t0, t1 = 1.0, z
            for _ in range(idx-1):
                t0, t1 = t1, 2*z*t1 - t0
            return t1

    def grad(self, x):
        g = np.zeros(self.n)
        for j in range(self.n):
            zj = 2.0*x[j] - 1.0
            for i in range(self.n):
                s_val = 0.0
                for k in range(self.n):
                    s_val += self._T(i, 2.0*x[k] - 1.0)
                g[j] += 2.0 * s_val * (2.0 * self._T_prime(i, zj))
        return g

    def _T_prime(self, idx, z):
        if idx == 0:
            return 0.0
        elif idx == 1:
            return 1.0
        else:
            return idx * self._U(idx-1, z)

    def _U(self, idx, z):
        if idx == 0:
            return 1.0
        elif idx == 1:
            return 2.0*z
        else:
            u0, u1 = 1.0, 2.0*z
            for _ in range(idx-1):
                u0, u1 = u1, 2*z*u1 - u0
            return u1


class Watson(TestFunction):
    """Watson function: f(x) = sum_{i=1}^{29} (sum_{j=2}^n (j-1)*x_j*t_i^{j-2}
       - (sum_{j=1}^n x_j*t_i^{j-1})^2 - 1)^2 + x_1^2"""
    def __init__(self, n=12):
        super().__init__("Watson", n, fstar=None)
        self.x0 = np.zeros(n)
        self._t = np.array([float(i)/29.0 for i in range(1, 30)])

    def eval(self, x):
        f = x[0]**2
        t_pow = np.ones((29, self.n))
        for j in range(self.n):
            if j > 0:
                t_pow[:, j] = t_pow[:, j-1] * self._t
        for i in range(29):
            s1 = 0.0
            for j in range(1, self.n):
                s1 += j * x[j] * t_pow[i, j-1]
            s2 = 0.0
            for j in range(self.n):
                s2 += x[j] * t_pow[i, j]
            f += (s1 - s2**2 - 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        g[0] = 2.0 * x[0]
        t_pow = np.ones((29, self.n))
        for j in range(self.n):
            if j > 0:
                t_pow[:, j] = t_pow[:, j-1] * self._t
        for i in range(29):
            s1 = 0.0
            for j in range(1, self.n):
                s1 += j * x[j] * t_pow[i, j-1]
            s2 = 0.0
            for j in range(self.n):
                s2 += x[j] * t_pow[i, j]
            ri = s1 - s2**2 - 1.0
            for j in range(self.n):
                if j == 0:
                    g[j] += 2.0 * ri * (-2.0 * s2 * t_pow[i, j])
                else:
                    g[j] += 2.0 * ri * (j * t_pow[i, j-1] - 2.0 * s2 * t_pow[i, j])
        return g


class Gulf(TestFunction):
    """Gulf research and development function, n=3"""
    def __init__(self):
        super().__init__("Gulf", 3, np.array([5.0, 2.5, 0.15]), fstar=0.0)
        self._t = np.array([i/100.0 for i in range(1, 100)])

    def eval(self, x):
        f = 0.0
        for i, ti in enumerate(self._t):
            yi = 25.0 + (-50.0*np.log(ti))**(2.0/3.0)
            fi = np.exp(-abs(yi - x[1])**x[2]/x[0]) - ti
            f += fi**2
        return f

    def grad(self, x):
        g = np.zeros(3)
        for ti in self._t:
            yi = 25.0 + (-50.0*np.log(ti))**(2.0/3.0)
            d = abs(yi - x[1])**x[2]
            e = np.exp(-d / x[0])
            fi = e - ti
            g[0] += 2.0 * fi * e * d / (x[0]**2)
            g[1] += 2.0 * fi * e * x[2] * abs(yi - x[1])**(x[2]-1) * np.sign(yi - x[1]) / x[0]
            g[2] += 2.0 * fi * e * (-d * np.log(abs(yi - x[1]) + 1e-15)) / x[0]
        return g


class Himmelblau(TestFunction):
    """Himmelblau function"""
    def __init__(self):
        super().__init__("Himmelblau", 2, np.array([1.0, 1.0]), fstar=0.0)

    def eval(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def grad(self, x):
        g1 = 4.0*x[0]*(x[0]**2 + x[1] - 11) + 2.0*(x[0] + x[1]**2 - 7)
        g2 = 2.0*(x[0]**2 + x[1] - 11) + 4.0*x[1]*(x[0] + x[1]**2 - 7)
        return np.array([g1, g2])


# ============================================================================
# Part 2: Modern Nonconvex Test Problems
# ============================================================================

class PhaseRetrieval(TestFunction):
    """Wirtinger flow: f(x) = (1/2m) * sum_i (|a_i^T x|^2 - b_i)^2 = (1/2m)*sum_i ((a_i^T x)^2 - b_i)^2"""
    def __init__(self, n=128, m=None, seed=42):
        super().__init__("Phase Retrieval", n, fstar=0.0)
        if m is None:
            m = 6 * n
        self._m = m
        rng = np.random.RandomState(seed)
        self.A = rng.randn(m, n)
        self.x_true = rng.randn(n)
        # b_i = (a_i^T x_true)^2
        self.b = (self.A @ self.x_true)**2
        self.x0 = rng.randn(n)

    def eval(self, x):
        Ax = self.A @ x
        residuals = Ax**2 - self.b
        return 0.5 * np.mean(residuals**2)

    def grad(self, x):
        Ax = self.A @ x
        residuals = Ax**2 - self.b
        return (2.0 / self._m) * (self.A.T @ (residuals * Ax))


class LowRankFactorization(TestFunction):
    """f(U, V) = ||UV^T - M||_F^2 for U, V in R^{d x r}"""
    def __init__(self, d=50, r=5, noise=0.01, seed=42):
        name = f"Low-Rank Fact (d={d}, r={r})"
        rng = np.random.RandomState(seed)
        self.d = d
        self.r = r
        self.U_true = rng.randn(d, r)
        self.V_true = rng.randn(d, r)
        self.M = self.U_true @ self.V_true.T + noise * rng.randn(d, d)
        self.x0 = rng.randn(2*d*r)
        super().__init__(name, 2*d*r, self.x0, fstar=0.0)

    def _unpack(self, x):
        n = self.d * self.r
        return x[:n].reshape(self.d, self.r), x[n:].reshape(self.d, self.r)

    def eval(self, x):
        U, V = self._unpack(x)
        diff = U @ V.T - self.M
        return 0.5 * np.sum(diff**2)

    def grad(self, x):
        U, V = self._unpack(x)
        diff = U @ V.T - self.M
        gU = (diff @ V).ravel()
        gV = (diff.T @ U).ravel()
        return np.concatenate([gU, gV])


class RobustRegression(TestFunction):
    """Tukey biweight robust regression"""
    def __init__(self, N=100, n=20, c=4.685, seed=42):
        name = f"Robust Reg (N={N}, n={n})"
        rng = np.random.RandomState(seed)
        self.X_data = rng.randn(N, n)
        beta_true = rng.randn(n)
        y = self.X_data @ beta_true + 0.1 * rng.randn(N)
        y[rng.choice(N, 5, replace=False)] += 10.0  # outliers
        self.y = y
        self.c = c
        self.x0 = np.zeros(n)
        super().__init__(name, n, self.x0, fstar=None)

    def eval(self, x):
        residuals = self.y - self.X_data @ x
        return np.sum(self._rho(residuals))

    def grad(self, x):
        residuals = self.y - self.X_data @ x
        psi_vals = self._psi(residuals)
        return -self.X_data.T @ psi_vals

    def _rho(self, u):
        c = self.c
        abs_u = np.abs(u)
        result = np.where(abs_u <= c,
                          c**2/6.0 * (1.0 - (1.0 - (abs_u/c)**2)**3),
                          c**2/6.0 * np.ones_like(u))
        return result

    def _psi(self, u):
        c = self.c
        abs_u = np.abs(u)
        result = np.where(abs_u <= c,
                          u * (1.0 - (abs_u/c)**2)**2,
                          np.zeros_like(u))
        return result


# ============================================================================
# Part 3: Stress Test Functions
# ============================================================================

class OscillatoryCurvature(TestFunction):
    """f(x) = sum_{i=1}^{n-1} [(x_i^2 + x_{i+1}^2)^2 - c_i*x_i] + eta*sum sin(omega*x_i)"""
    def __init__(self, n=50, eta=0.5, omega=20.0, seed=42):
        super().__init__(f"Oscillatory (n={n})", n, fstar=None)
        rng = np.random.RandomState(seed)
        self.c = rng.uniform(-2, 2, n-1)
        self.eta = eta
        self.omega = omega
        self.x0 = rng.randn(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i]**2 + x[i+1]**2)**2 - self.c[i]*x[i]
        f += self.eta * np.sum(np.sin(self.omega * x))
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 4.0*x[i]*(x[i]**2 + x[i+1]**2) - self.c[i]
            g[i+1] += 4.0*x[i+1]*(x[i]**2 + x[i+1]**2)
        g += self.eta * self.omega * np.cos(self.omega * x)
        return g


class FlatSteepAlternating(TestFunction):
    """Rosenbrock with multiplicative steepening"""
    def __init__(self, n=50):
        super().__init__(f"Flat-Steep Alt (n={n})", n, fstar=0.0)
        x0 = np.zeros(n)
        for i in range(0, n, 2):
            x0[i] = -1.2; x0[i+1] = 1.0
        self.x0 = x0

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 2):
            phi = 1.0 + 10.0 * np.exp(-x[i]**2 / 0.01)
            f += phi * (100.0*(x[i]**2 - x[i+1])**2 + (x[i] - 1.0)**2)
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 2):
            phi = 1.0 + 10.0 * np.exp(-x[i]**2 / 0.01)
            dphi = -2000.0 * x[i] * np.exp(-x[i]**2 / 0.01) / 0.01
            rb = 100.0*(x[i]**2 - x[i+1])**2 + (x[i] - 1.0)**2
            g[i] = dphi * rb + phi * (400.0*x[i]*(x[i]**2 - x[i+1]) + 2.0*(x[i] - 1.0))
            g[i+1] = phi * (-200.0*(x[i]**2 - x[i+1]))
        return g


class SaddleDense(TestFunction):
    """f(x) = 0.5*sum x_i^2 + sum_{i=1}^{n-2} x_i x_{i+1} x_{i+2}"""
    def __init__(self, n=50):
        super().__init__(f"Saddle-Dense (n={n})", n, fstar=0.0)
        self.x0 = 3.0 * np.ones(n)

    def eval(self, x):
        f = 0.5 * np.sum(x**2)
        for i in range(self.n - 2):
            f += x[i] * x[i+1] * x[i+2]
        return f

    def grad(self, x):
        g = x.copy()
        for i in range(self.n - 2):
            g[i] += x[i+1] * x[i+2]
            g[i+1] += x[i] * x[i+2]
            g[i+2] += x[i] * x[i+1]
        return g


class ExtremeConditionPenalty(TestFunction):
    """f(x) = 0.5*sum lambda_i*x_i^2 + mu*(sum x_i^2 - 1)^2, lambda_i exponential"""
    def __init__(self, n=50, mu=1000.0):
        super().__init__(f"Extreme Cond (n={n})", n, fstar=None)
        self.lambdas = 10.0**((np.arange(n))/(n-1))
        self.mu = mu
        self.x0 = np.ones(n) / np.sqrt(n)

    def eval(self, x):
        s = np.sum(x**2)
        return 0.5 * np.sum(self.lambdas * x**2) + self.mu * (s - 1.0)**2

    def grad(self, x):
        s = np.sum(x**2)
        return self.lambdas * x + 4.0 * self.mu * (s - 1.0) * x


# ============================================================================
# Problem Registry
# ============================================================================

def build_problem_suite(max_dim=200):
    """Build a comprehensive suite of test problems."""
    problems = []

    # Classical CUTEst problems
    for n in [10, 50, 200]:
        if n <= max_dim:
            problems.append(ExtendedRosenbrock(n))
            if n % 4 == 0:
                problems.append(ExtendedPowell(n))
            if n % 2 == 0:
                problems.append(ExtendedBeale(n))
            problems.append(ARWHEAD(n))
            problems.append(NONDIA(n))
            problems.append(PenaltyI(n))
            problems.append(PenaltyII(n))
            problems.append(BrownAlmostLinear(n))
            problems.append(Cosine(n))
            problems.append(BroydenTridiagonal(n))
            if n >= 5:
                problems.append(BDQRTIC(n))

    # Fixed-size problems
    problems.append(FreudensteinRoth())
    problems.append(HelicalValley())
    problems.append(Himmelblau())
    problems.append(Chebyquad(8))
    problems.append(Watson(12))
    problems.append(Gulf())

    # Modern nonconvex problems
    problems.append(PhaseRetrieval(64, seed=42))
    problems.append(PhaseRetrieval(128, seed=123))
    problems.append(LowRankFactorization(50, 5, seed=42))
    problems.append(LowRankFactorization(50, 10, seed=123))
    problems.append(RobustRegression(100, 20, seed=42))

    # Stress tests
    problems.append(OscillatoryCurvature(50, seed=42))
    problems.append(FlatSteepAlternating(50))
    problems.append(SaddleDense(50))
    problems.append(ExtremeConditionPenalty(50))

    return problems


if __name__ == "__main__":
    # Quick test
    problems = build_problem_suite(max_dim=50)
    print(f"Loaded {len(problems)} test problems")
    for p in problems[:5]:
        f0 = p.eval(p.x0)
        g0 = p.grad(p.x0)
        print(f"  {p.name:30s} n={p.n:4d} f(x0)={f0:.4e} ||g0||={norm(g0):.4e}")
