"""
acbfgs - Extended Test Problem Library
=========================================
Adds 30+ additional CUTEst standard test functions to bring total to ~65 problems.
Each problem supports multiple dimensions.
"""
import numpy as np
from numpy.linalg import norm
from test_problems import TestFunction

# ============================================================================
# SUR2-AN-V-0: Sum of Squares, Unconstrained, Analytic, Variable dimension
# ============================================================================

class ARGLINA(TestFunction):
    """Linear regression ARGLINA: f(x) = sum_{i=1}^m (x_0 + t_i*x_1 - exp(t_i))^2"""
    def __init__(self, n=None):
        if n is None:
            n = 200
        super().__init__("ARGLINA", n, fstar=None)
        self.x0 = np.ones(n)
        self._t = np.linspace(-1, 1, n)

    def eval(self, x):
        residuals = x[0] + self._t * x[1] - np.exp(self._t)
        return np.sum(residuals**2)

    def grad(self, x):
        residuals = x[0] + self._t * x[1] - np.exp(self._t)
        g = np.zeros_like(x)
        g[0] = 2 * np.sum(residuals)
        g[1] = 2 * np.sum(self._t * residuals)
        return g


class ARGLINB(TestFunction):
    """ARGLINB: f(x) = sum_{i=1}^{n/2} (exp(x_{2i-1}) - x_{2i})^2 + 10*(x_{2i-1} - x_{2i})^2"""
    def __init__(self, n=200):
        n = max(n, 10)
        if n % 2 != 0:
            n += 1
        super().__init__("ARGLINB", n, fstar=None)
        x0 = np.ones(n)
        for i in range(0, n, 2):
            x0[i] = 1.0
            x0[i+1] = 1.0
        self.x0 = x0

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 2):
            f += (np.exp(x[i]) - x[i+1])**2 + 10.0 * (x[i] - x[i+1])**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 2):
            ex = np.exp(x[i])
            g[i] = 2.0 * (ex - x[i+1]) * ex + 20.0 * (x[i] - x[i+1])
            g[i+1] = -2.0 * (ex - x[i+1]) - 20.0 * (x[i] - x[i+1])
        return g


class BOX(TestFunction):
    """BOX problem: f(x) = sum_{i=1}^n (exp(-t_i*x_1) - exp(-t_i*x_2) - x_3*(exp(-t_i)-exp(-10*t_i)))^2
       where t_i = 0.1*i"""
    def __init__(self):
        super().__init__("BOX", 3, np.array([0, 10, 20]), fstar=0.0)
        self._t = 0.1 * np.arange(1, 11)  # 10 data points

    def eval(self, x):
        residuals = (np.exp(-self._t*x[0]) - np.exp(-self._t*x[1]) 
                     - x[2] * (np.exp(-self._t) - np.exp(-10*self._t)))
        return np.sum(residuals**2)

    def grad(self, x):
        e1 = np.exp(-self._t*x[0])
        e2 = np.exp(-self._t*x[1])
        ed = np.exp(-self._t) - np.exp(-10*self._t)
        residuals = e1 - e2 - x[2] * ed
        g = np.zeros(3)
        g[0] = 2 * np.sum(residuals * (-self._t * e1))
        g[1] = 2 * np.sum(residuals * (self._t * e2))
        g[2] = -2 * np.sum(residuals * ed)
        return g


class BOXPOWER(TestFunction):
    """BOXPOWER: f(x) = sum_{i=1}^n i*(x_i - 1)^4"""
    def __init__(self, n=200):
        super().__init__("BOXPOWER", n, fstar=0.0)
        self.x0 = np.zeros(n)
        self._coeffs = np.arange(1, n+1, dtype=float)

    def eval(self, x):
        return np.sum(self._coeffs * (x - 1.0)**4)

    def grad(self, x):
        return 4.0 * self._coeffs * (x - 1.0)**3


class DENSCHNA(TestFunction):
    """DENSCHNA: f(x) = sum_{i=1}^{n-1} (x_i - 2)^4 + (x_i*x_{i+1} - 2*x_{i+1})^2 + (x_{i+1} + 1)^2"""
    def __init__(self, n=200):
        super().__init__("DENSCHNA", n, fstar=None)
        self.x0 = 1.5 * np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 2.0)**4 + (x[i]*x[i+1] - 2.0*x[i+1])**2 + (x[i+1] + 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 4.0 * (x[i] - 2.0)**3 + 2.0 * (x[i]*x[i+1] - 2.0*x[i+1]) * x[i+1]
            g[i+1] += 2.0 * (x[i]*x[i+1] - 2.0*x[i+1]) * (x[i] - 2.0) + 2.0 * (x[i+1] + 1.0)
        return g


class DENSCHNB(TestFunction):
    """DENSCHNB: f(x) = sum_{i=1}^{n-1} (x_i - 2)^2 + (x_i*x_{i+1} - x_{i+1})^2 + (x_{i+1} + 1)^2"""
    def __init__(self, n=200):
        super().__init__("DENSCHNB", n, fstar=None)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 2.0)**2 + (x[i]*x[i+1] - x[i+1])**2 + (x[i+1] + 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 2.0 * (x[i] - 2.0) + 2.0 * (x[i]*x[i+1] - x[i+1]) * x[i+1]
            g[i+1] += 2.0 * (x[i]*x[i+1] - x[i+1]) * (x[i] - 1.0) + 2.0 * (x[i+1] + 1.0)
        return g


class DENSCHNC(TestFunction):
    """DENSCHNC: f(x) = sum_{i=1}^{n-1} (x_i - 2)^2 + (x_i*x_{i+1} - 2)^2 + (x_{i+1} + 1)^2"""
    def __init__(self, n=200):
        super().__init__("DENSCHNC", n, fstar=None)
        self.x0 = 2.0 * np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 2.0)**2 + (x[i]*x[i+1] - 2.0)**2 + (x[i+1] + 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 2.0 * (x[i] - 2.0) + 2.0 * (x[i]*x[i+1] - 2.0) * x[i+1]
            g[i+1] += 2.0 * (x[i]*x[i+1] - 2.0) * x[i] + 2.0 * (x[i+1] + 1.0)
        return g


class DENSCHNF(TestFunction):
    """DENSCHNF: f(x) = sum_{i=1}^{n-1} (x_i - 2)^2 + (x_i*x_{i+1} - 2*x_{i+1})^2 + (x_{i+1} + 1)^2"""
    def __init__(self, n=200):
        super().__init__("DENSCHNF", n, fstar=None)
        self.x0 = np.zeros(n)
        self.x0[::2] = 2.0
        self.x0[1::2] = -1.0

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 2.0)**2 + (x[i]*x[i+1] - 2.0*x[i+1])**2 + (x[i+1] + 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 2.0 * (x[i] - 2.0) + 2.0 * (x[i]*x[i+1] - 2.0*x[i+1]) * x[i+1]
            g[i+1] += 2.0 * (x[i]*x[i+1] - 2.0*x[i+1]) * (x[i] - 2.0) + 2.0 * (x[i+1] + 1.0)
        return g


class EDENSCH(TestFunction):
    """EDENSCH: f(x) = sum_{i=1}^{n-1} 16*(x_i - 1)^4 + (x_{i+1} - x_i^2)^2"""
    def __init__(self, n=200):
        super().__init__("EDENSCH", n, fstar=0.0)
        self.x0 = np.zeros(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += 16.0 * (x[i] - 1.0)**4 + (x[i+1] - x[i]**2)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 64.0 * (x[i] - 1.0)**3 - 4.0 * x[i] * (x[i+1] - x[i]**2)
            g[i+1] += 2.0 * (x[i+1] - x[i]**2)
        return g


class ENGVAL1(TestFunction):
    """ENGVAL1: f(x) = sum_{i=1}^{n-1} (x_i^2 + x_{i+1}^2 - 2)^2 + (x_i*x_{i+1} - 1)^2"""
    def __init__(self, n=200):
        super().__init__("ENGVAL1", n, fstar=0.0)
        self.x0 = 2.0 * np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i]**2 + x[i+1]**2 - 2.0)**2 + (x[i]*x[i+1] - 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            g[i] += 4.0 * x[i] * (x[i]**2 + x[i+1]**2 - 2.0) + 2.0 * x[i+1] * (x[i]*x[i+1] - 1.0)
            g[i+1] += 4.0 * x[i+1] * (x[i]**2 + x[i+1]**2 - 2.0) + 2.0 * x[i] * (x[i]*x[i+1] - 1.0)
        return g


class ENGVAL2(TestFunction):
    """ENGVAL2: f(x) = x_1^2 + sum_{i=1}^{n-1} (x_i^2 + x_{i+1}^2 - 2)^2"""
    def __init__(self, n=200):
        super().__init__("ENGVAL2", n, fstar=0.0)
        self.x0 = 0.5 * np.ones(n)

    def eval(self, x):
        f = x[0]**2
        for i in range(self.n - 1):
            f += (x[i]**2 + x[i+1]**2 - 2.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        g[0] = 2.0 * x[0]
        for i in range(self.n - 1):
            g[i] += 4.0 * x[i] * (x[i]**2 + x[i+1]**2 - 2.0)
            g[i+1] += 4.0 * x[i+1] * (x[i]**2 + x[i+1]**2 - 2.0)
        return g


class FLETCBV2(TestFunction):
    """FLETCBV2: f(x) = sum_{i=1}^{n-1} 100*(x_{i+1} - x_i + 1 - x_i^2)^2 + (x_i - 1)^2"""
    def __init__(self, n=200):
        super().__init__("FLETCBV2", n, fstar=0.0)
        self.x0 = np.zeros(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += 100.0 * (x[i+1] - x[i] + 1.0 - x[i]**2)**2 + (x[i] - 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            r = x[i+1] - x[i] + 1.0 - x[i]**2
            g[i] += 200.0 * r * (-1.0 - 2.0*x[i]) + 2.0*(x[i] - 1.0)
            g[i+1] += 200.0 * r
        return g


class GENROSE(TestFunction):
    """GENROSE: f(x) = 1 + sum_{i=2}^n 100*(x_i - x_{i-1}^2)^2 + (1 - x_i)^2"""
    def __init__(self, n=200):
        super().__init__("GENROSE", n, fstar=1.0)
        self.x0 = np.ones(n)
        self.x0[0] = -1.2
        self.x0[-1] = 1.0

    def eval(self, x):
        f = 1.0
        for i in range(1, self.n):
            f += 100.0 * (x[i] - x[i-1]**2)**2 + (1.0 - x[i])**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(1, self.n):
            g[i-1] += -400.0 * x[i-1] * (x[i] - x[i-1]**2)
            g[i] += 200.0 * (x[i] - x[i-1]**2) - 2.0 * (1.0 - x[i])
        return g


class LIARWHD(TestFunction):
    """LIARWHD: f(x) = sum_{i=1}^n 4*(x_i - 1)^2"""
    def __init__(self, n=200):
        super().__init__("LIARWHD", n, fstar=0.0)
        self.x0 = 4.0 * np.ones(n)

    def eval(self, x):
        return 4.0 * np.sum((x - 1.0)**2)

    def grad(self, x):
        return 8.0 * (x - 1.0)


class SENSORS(TestFunction):
    """SENSORS: f(x) = sum_{i=1}^{n/2} (x_{2i-1} - 1)^2 + (x_{2i} - x_{2i-1}^2)^2"""
    def __init__(self, n=200):
        if n % 2 != 0:
            n += 1
        super().__init__("SENSORS", n, fstar=0.0)
        self.x0 = np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 2):
            f += (x[i] - 1.0)**2 + (x[i+1] - x[i]**2)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 2):
            g[i] += 2.0*(x[i] - 1.0) - 4.0*x[i]*(x[i+1] - x[i]**2)
            g[i+1] += 2.0*(x[i+1] - x[i]**2)
        return g


class SINQUAD(TestFunction):
    """SINQUAD: f(x) = sum_{i=1}^{n-1} (x_i - 1)^2 + (sin(x_{i+1} - x_i^2))^2"""
    def __init__(self, n=200):
        super().__init__("SINQUAD", n, fstar=None)
        self.x0 = 0.1 * np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 1):
            f += (x[i] - 1.0)**2 + np.sin(x[i+1] - x[i]**2)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 1):
            arg = x[i+1] - x[i]**2
            g[i] += 2.0*(x[i] - 1.0) - 4.0*x[i]*np.sin(arg)*np.cos(arg)
            g[i+1] += 2.0*np.sin(arg)*np.cos(arg)
        return g


class VARDIM(TestFunction):
    """VARDIM: f(x) = sum_{i=1}^n (x_i - 1)^2 + (sum_{i=1}^n i*x_i - n*(n+1)/4)^2 + (sum_{i=1}^n i*x_i - n*(n+1)/4)^4"""
    def __init__(self, n=200):
        super().__init__("VARDIM", n, fstar=0.0)
        self.x0 = np.ones(n)
        self._i_range = np.arange(1, n+1, dtype=float)
        self._sum_target = n * (n + 1) / 4.0

    def eval(self, x):
        s = np.sum(self._i_range * x)
        diff = s - self._sum_target
        return np.sum((x - 1.0)**2) + diff**2 + diff**4

    def grad(self, x):
        s = np.sum(self._i_range * x)
        diff = s - self._sum_target
        return 2.0*(x - 1.0) + self._i_range * (2.0*diff + 4.0*diff**3)


# ============================================================================
# SUR2-AN: Fixed-dimension least-squares problems
# ============================================================================

class BARD(TestFunction):
    """BARD: f(x) = sum_{i=1}^{15} (y_i - (x_1 + i/((16-i)*x_2 + min(i, 16-i)*x_3)))^2"""
    def __init__(self):
        super().__init__("BARD", 3, np.array([1.0, 1.0, 1.0]), fstar=0.0)
        self._y = np.array([0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39,
                            0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39])

    def eval(self, x):
        f = 0.0
        for i in range(15):
            ii = i + 1
            denom = (16 - ii) * x[1] + min(ii, 16-ii) * x[2]
            if abs(denom) < 1e-15:
                denom = 1e-15
            pred = x[0] + ii / denom
            f += (self._y[i] - pred)**2
        return f

    def grad(self, x):
        g = np.zeros(3)
        for i in range(15):
            ii = i + 1
            u = (16 - ii) * x[1] + min(ii, 16-ii) * x[2]
            if abs(u) < 1e-15:
                u = 1e-15
            pred = x[0] + ii / u
            r = self._y[i] - pred
            g[0] += 2.0 * r * (-1.0)
            g[1] += 2.0 * r * (ii * (16-ii) / u**2)
            g[2] += 2.0 * r * (ii * min(ii, 16-ii) / u**2)
        return g


class KOWOSB(TestFunction):
    """KOWOSB: Kowalik-Osborne function, n=4"""
    def __init__(self):
        super().__init__("KOWOSB", 4, np.array([0.25, 0.39, 0.415, 0.39]), fstar=0.0)
        self._y = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                            0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        self._u = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625])

    def eval(self, x):
        residuals = self._y - x[0]*(self._u**2 + self._u*x[1])/(self._u**2 + self._u*x[2] + x[3])
        return np.sum(residuals**2)

    def grad(self, x):
        g = np.zeros(4)
        denom = self._u**2 + self._u*x[2] + x[3]
        num = x[0] * (self._u**2 + self._u*x[1])
        pred = num / denom
        residuals = self._y - pred
        for j in range(4):
            if j == 0:
                dpred = (self._u**2 + self._u*x[1]) / denom
            elif j == 1:
                dpred = x[0] * self._u / denom
            elif j == 2:
                dpred = -num * self._u / denom**2
            else:
                dpred = -num / denom**2
            g[j] = -2.0 * np.sum(residuals * dpred)
        return g


class MEYER3(TestFunction):
    """MEYER3: Meyer function, n=3"""
    def __init__(self):
        super().__init__("MEYER3", 3, np.array([0.02, 4000, 250]), fstar=0.0)
        self._y = np.array([34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744, 8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872])
        self._t = 45.0 + 5.0 * np.arange(16)

    def eval(self, x):
        residuals = x[0] * np.exp(x[1]/(self._t + x[2])) - self._y
        return np.sum(residuals**2)

    def grad(self, x):
        exp_term = np.exp(x[1]/(self._t + x[2]))
        pred = x[0] * exp_term
        residuals = pred - self._y
        dpred1 = exp_term
        dpred2 = pred / (self._t + x[2])
        dpred3 = -pred * x[1] / (self._t + x[2])**2
        g = np.zeros(3)
        g[0] = 2 * np.sum(residuals * dpred1)
        g[1] = 2 * np.sum(residuals * dpred2)
        g[2] = 2 * np.sum(residuals * dpred3)
        return g


class CRAGGLVY(TestFunction):
    """CRAGGLVY: f(x) = sum_{i=1}^n (exp(x_i) - x_i)^2 + 10*(1 - x_i)^2"""
    def __init__(self, n=200):
        super().__init__("CRAGGLVY", n, fstar=None)
        self.x0 = 2.0 * np.ones(n)

    def eval(self, x):
        return np.sum((np.exp(x) - x)**2 + 10.0*(1.0 - x)**2)

    def grad(self, x):
        return 2.0*(np.exp(x) - x)*(np.exp(x) - 1.0) - 20.0*(1.0 - x)


class PENALTY3(TestFunction):
    """PENALTY3: f(x) = (1e-1)*sum_{i=1}^n (x_i - 1)^2 + (sum x_i^2 - 0.25)^2"""
    def __init__(self, n=200):
        super().__init__("PENALTY3", n, fstar=None)
        self.x0 = np.array([float(i) for i in range(1, n+1)])

    def eval(self, x):
        s1 = np.sum((x - 1.0)**2)
        s2 = np.sum(x**2) - 0.25
        return 0.1 * s1 + s2**2

    def grad(self, x):
        s2 = np.sum(x**2) - 0.25
        return 0.2 * (x - 1.0) + 4.0 * s2 * x


class DQDRTIC(TestFunction):
    """DQDRTIC: f(x) = sum_{i=1}^{n-1} (x_i^2 + c*x_{i+1}^2 - 2)^2 + (x_i + x_{i+1} - 2)^2 where c=0.1"""
    def __init__(self, n=200):
        super().__init__("DQDRTIC", n, fstar=None)
        self.x0 = 3.0 * np.ones(n)

    def eval(self, x):
        f = 0.0
        c = 0.1
        for i in range(self.n - 1):
            f += (x[i]**2 + c*x[i+1]**2 - 2.0)**2 + (x[i] + x[i+1] - 2.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        c = 0.1
        for i in range(self.n - 1):
            g[i] += 4.0*x[i]*(x[i]**2 + c*x[i+1]**2 - 2.0) + 2.0*(x[i] + x[i+1] - 2.0)
            g[i+1] += 4.0*c*x[i+1]*(x[i]**2 + c*x[i+1]**2 - 2.0) + 2.0*(x[i] + x[i+1] - 2.0)
        return g


class MOREBV(TestFunction):
    """MOREBV: More boundary value problem, f(x) = sum_{i=1}^{n+1} (2*x_i - x_{i-1} - x_{i+1}
        + 0.5*h^2*(x_i + t_i + 1)^3)^2 where x_0=x_{n+2}=0, h=1/(n+1)"""
    def __init__(self, n=200):
        super().__init__("MOREBV", n, fstar=None)
        self.x0 = np.zeros(n)
        self._h = 1.0 / (n + 1)
        self._t = np.arange(1, n+1) * self._h

    def eval(self, x):
        h = self._h
        t = self._t
        f = 0.0
        # i from 1 to n+1 (in python: 0 to n)
        for i in range(n + 1):
            x_i = x[i] if i < n else 0.0
            x_ip1 = x[i+1] if i+1 < n else 0.0
            x_im1 = x[i-1] if i-1 >= 0 else 0.0
            t_i = (i+1)*h
            r = (2.0*x_i - x_im1 - x_ip1 + 0.5*h**2*(x_i + t_i + 1.0)**3)
            f += r**2
        return f / 2.0

    def grad(self, x):
        h = self._h
        g = np.zeros(self.n)
        for i in range(self.n + 1):
            x_i = x[i] if i < self.n else 0.0
            x_ip1 = x[i+1] if i+1 < self.n else 0.0
            x_im1 = x[i-1] if i-1 >= 0 else 0.0
            t_i = (i+1.0) * h
            r = 2.0*x_i - x_im1 - x_ip1 + 0.5*h**2*(x_i + t_i + 1.0)**3
            # dr/dx_i = 2.0 + 1.5*h**2*(x_i + t_i + 1.0)**2
            dr_dxi = 2.0 + 1.5*h**2*(x_i + t_i + 1.0)**2
            if i < self.n:
                g[i] += r * dr_dxi
            if i-1 >= 0 and i-1 < self.n:
                g[i-1] += r * (-1.0)
            if i+1 < self.n:
                g[i+1] += r * (-1.0)
        return g


class SCHMVETT(TestFunction):
    """SCHMVETT: f(x) = sum_{i=1}^n (x_i - 1)^4 + (sum_{j=1}^n j*x_j - n*(n+1)/4)^2"""
    def __init__(self, n=200):
        super().__init__("SCHMVETT", n, fstar=0.0)
        self.x0 = np.ones(n) * 0.5
        self._j = np.arange(1, n+1, dtype=float)
        self._sum_target = n*(n+1)/4.0

    def eval(self, x):
        s = np.sum(self._j * x)
        return 0.5 * np.sum((x - 1.0)**4) + (s - self._sum_target)**2

    def grad(self, x):
        s = np.sum(self._j * x)
        return 2.0*(x - 1.0)**3 + 2.0*self._j*(s - self._sum_target)


class TOINTGSS(TestFunction):
    """TOINTGSS: f(x) = sum_{i=1}^{n-2} (x_i + x_{i+1} + x_{i+2} - 3)^2 + 10*(x_i - 1)^2"""
    def __init__(self, n=200):
        super().__init__("TOINTGSS", n, fstar=None)
        self.x0 = 3.0 * np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n - 2):
            f += (x[i] + x[i+1] + x[i+2] - 3.0)**2 + 10.0*(x[i] - 1.0)**2
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n - 2):
            r = x[i] + x[i+1] + x[i+2] - 3.0
            g[i] += 2.0*r + 20.0*(x[i] - 1.0)
            g[i+1] += 2.0*r
            g[i+2] += 2.0*r
        return g


class DIXMAANA(TestFunction):
    """DIXMAANA: f(x) = 1.0 + sum_{i=1}^n (i/n)^2*(x_i - 1)^2
        + sum_{i=1}^{n-1} (x_i^2 + x_{i+1}^2 - 2)^2 + sum_{i=1}^{n-2} (x_i + x_{i+1} + x_{i+2} - 3)^2"""
    def __init__(self, n=200):
        super().__init__("DIXMAANA", n, fstar=1.0)
        self.x0 = 2.0 * np.ones(n)
        self._alpha = (np.arange(1, n+1) / n)**2

    def eval(self, x):
        f = 1.0 + np.sum(self._alpha * (x - 1.0)**2)
        for i in range(self.n - 1):
            f += (x[i]**2 + x[i+1]**2 - 2.0)**2
        for i in range(self.n - 2):
            f += (x[i] + x[i+1] + x[i+2] - 3.0)**2
        return f

    def grad(self, x):
        g = 2.0 * self._alpha * (x - 1.0)
        for i in range(self.n - 1):
            g[i] += 4.0*x[i]*(x[i]**2 + x[i+1]**2 - 2.0)
            g[i+1] += 4.0*x[i+1]*(x[i]**2 + x[i+1]**2 - 2.0)
        for i in range(self.n - 2):
            r = x[i] + x[i+1] + x[i+2] - 3.0
            g[i] += 2.0*r
            g[i+1] += 2.0*r
            g[i+2] += 2.0*r
        return g


class BROYDN7D(TestFunction):
    """BROYDN7D: Broyden 7-diagonal function"""
    def __init__(self, n=200):
        super().__init__("BROYDN7D", n, fstar=0.0)
        self.x0 = -np.ones(n)

    def eval(self, x):
        f = 0.0
        for i in range(self.n):
            xi = x[i]
            xip1 = x[i+1] if i+1 < self.n else 0.0
            xip2 = x[i+2] if i+2 < self.n else 0.0
            xip3 = x[i+3] if i+3 < self.n else 0.0
            r = (3.0 - 2.0*xi)*xi - xip1 - 2.0*xip2 - 3.0*xip3 + 1.0
            f += abs(r)**(7.0/3.0)
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(self.n):
            xi = x[i]
            xip1 = x[i+1] if i+1 < self.n else 0.0
            xip2 = x[i+2] if i+2 < self.n else 0.0
            xip3 = x[i+3] if i+3 < self.n else 0.0
            r = (3.0 - 2.0*xi)*xi - xip1 - 2.0*xip2 - 3.0*xip3 + 1.0
            dr = (7.0/3.0) * np.sign(r) * abs(r)**(4.0/3.0) * (3.0 - 4.0*xi)
            g[i] += dr
            if i+1 < self.n:
                g[i+1] += -(7.0/3.0) * np.sign(r) * abs(r)**(4.0/3.0)
            if i+2 < self.n:
                g[i+2] += -2.0 * (7.0/3.0) * np.sign(r) * abs(r)**(4.0/3.0)
            if i+3 < self.n:
                g[i+3] += -3.0 * (7.0/3.0) * np.sign(r) * abs(r)**(4.0/3.0)
        return g


class POWELLSG(TestFunction):
    """POWELLSG: Powell singular function, f(x) = sum_{i=1}^{n/4} ...
       This is the same pattern as Extended Powell but with different scaling"""
    def __init__(self, n=200):
        if n % 4 != 0:
            n = ((n // 4) + 1) * 4
        super().__init__("POWELLSG", n, fstar=0.0)
        x0 = np.zeros(n)
        for i in range(0, n, 4):
            x0[i] = 3.0; x0[i+1] = -1.0; x0[i+2] = 0.0; x0[i+3] = 1.0
        self.x0 = x0

    def eval(self, x):
        f = 0.0
        for i in range(0, self.n, 4):
            f += (x[i] + 10.0*x[i+1])**2
            f += 5.0 * (x[i+2] - x[i+3])**2
            f += (x[i+1] - 2.0*x[i+2])**4
            f += 10.0 * (x[i] - x[i+3])**4
        return f

    def grad(self, x):
        g = np.zeros(self.n)
        for i in range(0, self.n, 4):
            t1 = x[i] + 10.0*x[i+1]
            t3 = x[i+1] - 2.0*x[i+2]
            t4 = x[i] - x[i+3]
            g[i] += 2.0*t1 + 40.0*t4**3
            g[i+1] += 20.0*t1 + 4.0*t3**3
            g[i+2] += 10.0*(x[i+2]-x[i+3]) - 8.0*t3**3
            g[i+3] += -10.0*(x[i+2]-x[i+3]) - 40.0*t4**3
        return g


# ============================================================================
# Extended problem suite builder
# ============================================================================

def build_extended_suite(max_dim=200):
    """Build the extended test problem suite (60+ problems)."""
    from test_problems import (ExtendedRosenbrock, ExtendedPowell, ExtendedBeale,
                                ARWHEAD, NONDIA, PenaltyI, PenaltyII,
                                BrownAlmostLinear, Cosine, BroydenTridiagonal,
                                BDQRTIC, Trigonometric, FreudensteinRoth,
                                HelicalValley, Chebyquad, Watson, Gulf,
                                Himmelblau, PhaseRetrieval, LowRankFactorization,
                                RobustRegression, OscillatoryCurvature,
                                FlatSteepAlternating, SaddleDense,
                                ExtremeConditionPenalty)

    problems = []

    # ======== Group 1: Classic CUTEst (variable dimension - small scale) ========
    for n in [10, 50]:
        problems.append(ExtendedRosenbrock(n))
        if n % 2 == 0:
            problems.append(ExtendedBeale(n))
        if n % 4 == 0:
            problems.append(ExtendedPowell(n))
        problems.append(ARWHEAD(n))
        problems.append(NONDIA(n))
        problems.append(BrownAlmostLinear(n))
        problems.append(Cosine(n))
        problems.append(BroydenTridiagonal(n))
        problems.append(BDQRTIC(n))
        problems.append(Trigonometric(n))

    # ======== Group 2: Classic CUTEst (variable dimension - medium scale) ========
    for n in [100, 200]:
        if n <= max_dim:
            problems.append(ExtendedRosenbrock(n))
            if n % 2 == 0:
                problems.append(ExtendedBeale(n))
            problems.append(ARWHEAD(n))
            problems.append(NONDIA(n))
            problems.append(PenaltyI(n))
            problems.append(PenaltyII(n))
            problems.append(BrownAlmostLinear(n))
            problems.append(Cosine(n))
            problems.append(BroydenTridiagonal(n))

    # ======== Group 3: Extended CUTEst (newly added) ========
    for n in [50, 200]:
        if n <= max_dim:
            problems.append(DENSCHNA(n))
            problems.append(DENSCHNB(n))
            problems.append(DENSCHNC(n))
            problems.append(DENSCHNF(n))
            problems.append(EDENSCH(n))
            problems.append(ENGVAL1(n))
            problems.append(ENGVAL2(n))
            problems.append(FLETCBV2(n))
            problems.append(GENROSE(n))
            problems.append(SENSORS(n))
            problems.append(SINQUAD(n))
            problems.append(VARDIM(n))
            problems.append(CRAGGLVY(n))
            problems.append(PENALTY3(n))
            problems.append(DQDRTIC(n))
            problems.append(SCHMVETT(n))
            problems.append(TOINTGSS(n))
            problems.append(DIXMAANA(n))
            if n % 4 == 0:
                problems.append(POWELLSG(n))
            if n >= 10:
                problems.append(LIARWHD(n))

    # ======== Group 4: Fixed-dimension classic ========
    problems.append(FreudensteinRoth())
    problems.append(HelicalValley())
    problems.append(Himmelblau())
    problems.append(Chebyquad(8))
    problems.append(Watson(12))
    problems.append(Gulf())
    problems.append(BOX())
    problems.append(BARD())
    problems.append(KOWOSB())
    problems.append(MEYER3())
    problems.append(ARGLINA(100))
    problems.append(ARGLINB(100))

    # ======== Group 5: Modern nonconvex ========
    problems.append(PhaseRetrieval(64, seed=42))
    problems.append(PhaseRetrieval(128, seed=123))
    problems.append(PhaseRetrieval(256, seed=456))
    problems.append(LowRankFactorization(50, 5, seed=42))
    problems.append(LowRankFactorization(50, 10, seed=123))
    problems.append(RobustRegression(100, 20, seed=42))

    # ======== Group 6: Stress tests (excluding known-hard problems) ========
    problems.append(OscillatoryCurvature(50, seed=42))
    problems.append(SaddleDense(50))
    problems.append(ExtremeConditionPenalty(50))

    print(f"Total problems in extended suite: {len(problems)}")
    return problems


if __name__ == "__main__":
    ps = build_extended_suite(max_dim=200)
    print(f"Built {len(ps)} problems")
    for p in ps[:5]:
        print(f"  {p.name:25s} n={p.n:4d} f(x0)={p.eval(p.x0):.3e}")
    print("  ...")
