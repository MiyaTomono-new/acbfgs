"""
acbfgs - Batch 3 Test Functions
=================================
Adds 20+ more CUTEst standard test functions to reach 80+ problem instances.
"""
import numpy as np
from numpy.linalg import norm
from test_problems import TestFunction

# ============================================================================
# Batch 3: Additional SUR2 / OUR2 / QUR2 functions
# ============================================================================

class BROWNAL(TestFunction):
    """BROWNAL: f(x) = sum_{i=1}^{n-1} (x_i - 3)^2 + (x_n - 1)^2 * (sum x_i - n)^2"""
    def __init__(self, n=200):
        super().__init__("BROWNAL", n, fstar=0.0)
        self.x0 = 0.5 * np.ones(n)

    def eval(self, x):
        f = np.sum((x[:-1] - 3.0)**2) + (x[-1] - 1.0)**2 * (np.sum(x) - self.n)**2
        return f

    def grad(self, x):
        s = np.sum(x)
        g = np.zeros(self.n)
        g[:-1] = 2.0 * (x[:-1] - 3.0) + 2.0 * (x[-1] - 1.0)**2 * s
        g[-1] = 2.0 * (x[-1] - 1.0) * s**2
        # Correction: derivative of (x_n-1)^2 * (sum x - n)^2
        diff = s - self.n
        xn1 = x[-1] - 1.0
        g[:-1] = 2.0 * (x[:-1] - 3.0) + 2.0 * xn1**2 * diff
        g[-1] = 2.0 * xn1 * diff**2
        return g


class DIXMAANB(TestFunction):
    """DIXMAANB: same as DIXMAANA but with alpha=1, beta=i^2/n^2"""
    def __init__(self, n=200):
        super().__init__("DIXMAANB", n, fstar=1.0)
        self.x0 = 2.0 * np.ones(n)
        self._beta = (np.arange(1, n+1) / n)**2

    def eval(self, x):
        f = 1.0 + np.sum((x - 1.0)**2)
        for i in range(self.n - 1):
            f += self._beta[i] * (x[i]**2 + x[i+1]**2 - 2.0)**2
        for i in range(self.n - 2):
            f += (x[i] + x[i+1] + x[i+2] - 3.0)**2
        return f

    def grad(self, x):
        g = 2.0 * (x - 1.0)
        for i in range(self.n - 1):
            g[i] += 4.0 * self._beta[i] * x[i] * (x[i]**2 + x[i+1]**2 - 2.0)
            g[i+1] += 4.0 * self._beta[i] * x[i+1] * (x[i]**2 + x[i+1]**2 - 2.0)
        for i in range(self.n - 2):
            r = x[i] + x[i+1] + x[i+2] - 3.0
            g[i] += 2.0 * r
            g[i+1] += 2.0 * r
            g[i+2] += 2.0 * r
        return g


class EG2(TestFunction):
    """EG2: f(x) = sum_{i=1}^{n-1} sin(x_i + x_{i+1} - 1) + (x_i - x_{i+1})^2
        + 1.5*x_i - x_{i+1} + 1"""
    def __init__(self, n=200):
        super().__init__("EG2", n, fstar=None)
        self.x0 = np.zeros(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += np.sin(x[i] + x[i+1] - 1.0) + (x[i] - x[i+1])**2 + 1.5*x[i] - x[i+1] + 1.0
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            c = np.cos(x[i] + x[i+1] - 1.0)
            g[i] += c + 2.0*(x[i] - x[i+1]) + 1.5
            g[i+1] += c - 2.0*(x[i] - x[i+1]) - 1.0
        return g


class FMINSRF2(TestFunction):
    """FMINSRF2: f(x) = sum_{i=1}^{n-1} (x_i^2 + x_{i+1}^2 - 2)^2"""
    def __init__(self, n=200):
        super().__init__("FMINSRF2", n, fstar=0.0)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i]**2 + x[i+1]**2 - 2.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 4.0 * x[i] * (x[i]**2 + x[i+1]**2 - 2.0)
            g[i+1] += 4.0 * x[i+1] * (x[i]**2 + x[i+1]**2 - 2.0)
        return g


class HILBERTA(TestFunction):
    """HILBERTA: f(x) = sum_{i=1}^n sum_{j=1}^n x_i * x_j / (i+j-1) - sum_i x_i"""
    def __init__(self, n=200):
        super().__init__("HILBERTA", n, fstar=None)
        self.x0 = np.ones(n)
        i_idx = np.arange(1, n+1).reshape(-1, 1)
        j_idx = np.arange(1, n+1).reshape(1, -1)
        self._H = 1.0 / (i_idx + j_idx - 1)

    def eval(self, x):
        return 0.5 * x.dot(self._H @ x) - np.sum(x)

    def grad(self, x):
        return self._H @ x - np.ones(self.n)


class LOGROS(TestFunction):
    """LOGROS: f(x) = sum_{i=1}^{n/2} log(1 + (x_{2i-1} - x_{2i})^2 + (1 - x_{2i-1})^2)"""
    def __init__(self, n=200):
        if n % 2 != 0:
            n += 1
        super().__init__("LOGROS", n, fstar=0.0)
        self.x0 = np.zeros(n)
        for i in range(0, n, 2):
            self.x0[i] = -1.2
            self.x0[i+1] = 1.0

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 2):
            f += np.log(1.0 + (x[i] - x[i+1])**2 + (1.0 - x[i])**2)
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 2):
            d = 1.0 + (x[i] - x[i+1])**2 + (1.0 - x[i])**2
            g[i] += (2.0*(x[i] - x[i+1]) - 2.0*(1.0 - x[i])) / d
            g[i+1] += -2.0*(x[i] - x[i+1]) / d
        return g


class MARATOSB(TestFunction):
    """MARATOSB: Maratos function, n=2"""
    def __init__(self):
        super().__init__("MARATOSB", 2, np.array([1.1, 0.1]), fstar=None)

    def eval(self, x):
        return x[0] + 100.0 * (x[0]**2 + x[1]**2 - 1.0)**2

    def grad(self, x):
        c = x[0]**2 + x[1]**2 - 1.0
        return np.array([1.0 + 400.0 * x[0] * c, 400.0 * x[1] * c])


class NONSCOMP(TestFunction):
    """NONSCOMP: f(x) = sum_{i=1}^{n-1} (x_i - 1)^2 + (x_i - cos(x_{i+1}))^2"""
    def __init__(self, n=200):
        super().__init__("NONSCOMP", n, fstar=None)
        self.x0 = 3.0 * np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 1.0)**2 + (x[i] - np.cos(x[i+1]))**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 2.0*(x[i] - 1.0) + 2.0*(x[i] - np.cos(x[i+1]))
            g[i+1] += 2.0*(x[i] - np.cos(x[i+1])) * np.sin(x[i+1])
        return g


class PROBPENL(TestFunction):
    """PROBPENL: f(x) = sum_{i=1}^{n-1} (x_i - 1)^2 + (x_i^2 + x_{i+1}^2 - 1)^2"""
    def __init__(self, n=200):
        super().__init__("PROBPENL", n, fstar=None)
        self.x0 = 0.5 * np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 1.0)**2 + (x[i]**2 + x[i+1]**2 - 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 2.0*(x[i] - 1.0) + 4.0*x[i]*(x[i]**2 + x[i+1]**2 - 1.0)
            g[i+1] += 4.0*x[i+1]*(x[i]**2 + x[i+1]**2 - 1.0)
        return g


class SINEALI(TestFunction):
    """SINEALI: f(x) = sum_{i=1}^{n-1} (sin(x_i - 1) - 2*x_i + x_{i+1})^2"""
    def __init__(self, n=200):
        super().__init__("SINEALI", n, fstar=None)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            r = np.sin(x[i] - 1.0) - 2.0*x[i] + x[i+1]
            f += r**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            r = np.sin(x[i] - 1.0) - 2.0*x[i] + x[i+1]
            g[i] += 2.0 * r * (np.cos(x[i] - 1.0) - 2.0)
            g[i+1] += 2.0 * r
        return g


class STRATEC(TestFunction):
    """STRATEC: n=10, f(x) = sum_{i=1}^{10} (x_i - i)^2 + (sum_i x_i - 55)^2"""
    def __init__(self):
        super().__init__("STRATEC", 10, np.zeros(10), fstar=0.0)

    def eval(self, x):
        return 0.5 * np.sum((x - np.arange(1, 11))**2) + 0.5 * (np.sum(x) - 55.0)**2

    def grad(self, x):
        return (x - np.arange(1, 11)) + (np.sum(x) - 55.0) * np.ones(10)


class TESTQUAD(TestFunction):
    """TESTQUAD: f(x) = sum_{i=1}^{n-1} (x_i - x_{i+1})^2 + (1 - x_1)^2"""
    def __init__(self, n=200):
        super().__init__("TESTQUAD", n, fstar=0.0)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = (1.0 - x[0])**2
        for i in range(self.n - 1):
            f += (x[i] - x[i+1])**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        g[0] = -2.0*(1.0 - x[0]) + 2.0*(x[0] - x[1])
        for i in range(1, self.n - 1):
            g[i] = 2.0*(x[i] - x[i-1]) - 2.0*(x[i+1] - x[i])
        g[-1] = -2.0*(x[-1] - x[-2])
        return g


class TOINTPSP(TestFunction):
    """TOINTPSP: f(x) = sum_{i=1}^{n-1} (x_i^2 + x_{i+1}^2 + x_i*x_{i+1} - 1)^2 + (x_i - 1)^2"""
    def __init__(self, n=200):
        super().__init__("TOINTPSP", n, fstar=None)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i]**2 + x[i+1]**2 + x[i]*x[i+1] - 1.0)**2 + (x[i] - 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            r = x[i]**2 + x[i+1]**2 + x[i]*x[i+1] - 1.0
            g[i] += 2.0*r*(2.0*x[i] + x[i+1]) + 2.0*(x[i] - 1.0)
            g[i+1] += 2.0*r*(2.0*x[i+1] + x[i])
        return g


class ZANGWIL2(TestFunction):
    """ZANGWIL2: f(x) = (x_1 - x_2 + x_3)^2 + (-x_1 + x_2 + x_3)^2 + (x_1 + x_2 - x_3)^2"""
    def __init__(self):
        super().__init__("ZANGWIL2", 3, np.array([100.0, -1.0, 2.5]), fstar=0.0)

    def eval(self, x):
        t1 = x[0] - x[1] + x[2]
        t2 = -x[0] + x[1] + x[2]
        t3 = x[0] + x[1] - x[2]
        return t1**2 + t2**2 + t3**2

    def grad(self, x):
        t1 = x[0] - x[1] + x[2]
        t2 = -x[0] + x[1] + x[2]
        t3 = x[0] + x[1] - x[2]
        return np.array([2*t1 - 2*t2 + 2*t3, -2*t1 + 2*t2 + 2*t3, 2*t1 + 2*t2 - 2*t3])


class DENSCHNE(TestFunction):
    """DENSCHNE: f(x) = sum_{i=1}^{n-1} (x_i - 2)^2 + (x_i - 2*x_{i+1})^2 + (2*x_i - x_{i+1})^2"""
    def __init__(self, n=200):
        super().__init__("DENSCHNE", n, fstar=None)
        self.x0 = np.zeros(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 2.0)**2 + (x[i] - 2.0*x[i+1])**2 + (2.0*x[i] - x[i+1])**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 2.0*(x[i] - 2.0) + 2.0*(x[i] - 2.0*x[i+1]) + 4.0*(2.0*x[i] - x[i+1])
            g[i+1] += -4.0*(x[i] - 2.0*x[i+1]) - 2.0*(2.0*x[i] - x[i+1])
        return g


class Hager(TestFunction):
    """Hager function: f(x) = sum_{i=1}^{n-1} (x_i - 2*sin(x_i + x_{i+1} - 1))^2 + (x_i - x_{i+1})^2"""
    def __init__(self, n=200):
        super().__init__("Hager", n, fstar=None)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 2.0*np.sin(x[i] + x[i+1] - 1.0))**2 + (x[i] - x[i+1])**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            s = x[i] + x[i+1] - 1.0
            r = x[i] - 2.0*np.sin(s)
            g[i] += 2.0*r*(1.0 - 2.0*np.cos(s)) + 2.0*(x[i] - x[i+1])
            g[i+1] += 2.0*r*(-2.0*np.cos(s)) - 2.0*(x[i] - x[i+1])
        return g


class TOINTGOR(TestFunction):
    """TOINTGOR: n=50, f(x) = sum_{i=1}^{49} (x_i - x_{i+1} + 1 - 2*x_i^2)^2"""
    def __init__(self, n=50):
        super().__init__("TOINTGOR", n, fstar=None)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - x[i+1] + 1.0 - 2.0*x[i]**2)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            r = x[i] - x[i+1] + 1.0 - 2.0*x[i]**2
            g[i] += 2.0*r*(1.0 - 4.0*x[i])
            g[i+1] += -2.0*r
        return g


class Biggs6(TestFunction):
    """BIGGS6: f(x) = sum_{i=1}^{13} (x_3*t_i^2 + x_2*t_i + x_1 - exp(-t_i) - 5*exp(-10*t_i))^2"""
    def __init__(self):
        super().__init__("BIGGS6", 6, np.ones(6), fstar=0.0)
        self._t = 0.1 * np.arange(1, 14)

    def eval(self, x):
        residuals = x[2]*self._t**2 + x[1]*self._t + x[0] - np.exp(-self._t) - 5.0*np.exp(-10.0*self._t)
        return np.sum(residuals**2)

    def grad(self, x):
        residuals = x[2]*self._t**2 + x[1]*self._t + x[0] - np.exp(-self._t) - 5.0*np.exp(-10.0*self._t)
        g = np.zeros(6)
        g[0] = 2.0 * np.sum(residuals)
        g[1] = 2.0 * np.sum(residuals * self._t)
        g[2] = 2.0 * np.sum(residuals * self._t**2)
        return g


# ============================================================================
# Build the full 80+ instance suite
# ============================================================================

def build_full_suite(max_dim=200):
    """Build 80+ problem instances for OMS-quality benchmarking."""
    from test_problems_extended import build_extended_suite
    problems = build_extended_suite(max_dim=max_dim)

    # Add new problems (variable dimension)
    for n in [50, 200]:
        if n <= max_dim:
            problems.append(BROWNAL(n))
            problems.append(DIXMAANB(n))
            problems.append(EG2(n))
            problems.append(FMINSRF2(n))
            problems.append(LOGROS(n))
            problems.append(NONSCOMP(n))
            problems.append(PROBPENL(n))
            problems.append(SINEALI(n))
            problems.append(TESTQUAD(n))
            problems.append(TOINTPSP(n))
            problems.append(DENSCHNE(n))
            problems.append(Hager(n))
            if n >= 50:
                problems.append(HILBERTA(min(n, 100)))

    # Add fixed-dimension problems
    problems.append(MARATOSB())
    problems.append(ZANGWIL2())
    problems.append(STRATEC())
    problems.append(Biggs6())
    problems.append(TOINTGOR(50))

    print(f"Full suite: {len(problems)} problem instances")
    return problems


if __name__ == "__main__":
    ps = build_full_suite(200)
    print(f"Built {len(ps)} instances")
