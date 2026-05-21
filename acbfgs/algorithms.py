"""
acbfgs - Adaptive Cautious BFGS Methods
========================================
Algorithm implementations: AC-BFGS, BFGS, Li-Fukushima CBFGS,
Powell Damped BFGS, L-BFGS, Gradient Descent.
"""
import numpy as np
from numpy.linalg import norm, solve
from scipy.linalg import eigh
import time

# ============================================================================
# Wolfe Line Search
# ============================================================================

def wolfe_line_search(f, grad_f, xk, dk, gk, alpha0=1.0, c1=1e-4, c2=0.9,
                       max_iters=50, rho=0.5):
    """Wolfe line search: satisfies sufficient decrease (Armijo) and curvature condition."""
    fk = f(xk)
    dphi0 = gk.dot(dk)
    if dphi0 >= 0:
        return 0.0, False  # not a descent direction

    alpha = alpha0
    alpha_min = 1e-20

    for i in range(max_iters):
        x_new = xk + alpha * dk
        f_new = f(x_new)
        # Check for NaN/Inf
        if not np.isfinite(f_new):
            alpha *= rho
            if alpha < alpha_min:
                return alpha, False
            continue

        # Armijo condition
        if f_new > fk + c1 * alpha * dphi0:
            alpha *= rho
            if alpha < alpha_min:
                return alpha, False
            continue

        # Curvature condition
        g_new = grad_f(x_new)
        if not np.all(np.isfinite(g_new)):
            alpha *= rho
            if alpha < alpha_min:
                return alpha, False
            continue
            
        if g_new.dot(dk) < c2 * dphi0:
            if f_new < fk:
                if alpha < 1e-4:
                    return alpha, True
                alpha = min(alpha / rho, 100.0)
                if alpha > 100.0:
                    return alpha0, True
                continue
            alpha *= rho
            if alpha < alpha_min:
                return alpha, False
            continue

        return alpha, True

    return alpha, False


# ============================================================================
# BFGS Update (inverse form)
# ============================================================================

def bfgs_update_h(H, s, y):
    """BFGS update for inverse Hessian H.
    H_{k+1} = (I - sy^T/y^Ts) H (I - ys^T/y^Ts) + ss^T/y^Ts
    """
    yTs = y.dot(s)
    if yTs <= 1e-15:
        return H  # skip
    rho = 1.0 / yTs
    I = np.eye(H.shape[0])
    V = I - rho * np.outer(s, y)
    H_new = V @ H @ V.T + rho * np.outer(s, s)
    return H_new


def bfgs_update_b(B, s, y):
    """BFGS update for Hessian approximation B.
    B_{k+1} = B - (B s s^T B)/(s^T B s) + (y y^T)/(y^T s)
    """
    yTs = y.dot(s)
    if yTs <= 1e-15:
        return B
    Bs = B @ s
    B_new = B - np.outer(Bs, Bs) / (s.dot(Bs)) + np.outer(y, y) / yTs
    return B_new


# ============================================================================
# H-condition number estimation (for condition monitoring)
# ============================================================================

def estimate_condition_number(H):
    """Estimate condition number of H (or compute exactly if cheap)."""
    n = H.shape[0]
    if n <= 500:
        try:
            eigs = eigh(H, eigvals_only=True, subset_by_index=[0, n-1])
            if eigs[0] <= 0:
                return 1e15
            return eigs[-1] / eigs[0]
        except:
            return norm(H, 'fro') * norm(np.linalg.inv(H), 'fro')
    else:
        return norm(H, 'fro') * norm(solve(H, np.eye(n), assume_a='pos'), 'fro')


# ============================================================================
# Solver Implementations
# ============================================================================

def acbfgs(f, grad_f, x0, max_iter=10000, tol=1e-6, 
           H0=None, alpha_scheme='default',
           eps0=None, eps_min=1e-6, eps_max=1e-2,
           rho_down=0.9, rho_up=1.2, m1=3, m3=2,
           tau=1e-8, alpha_min=1e-12, kappa_max=1e10, c_min=1e-6,
           callback=None, **kwargs):
    """
    AC-BFGS: Adaptive Cautious BFGS.
    
    Feedback signals:
      1. Sustained objective decrease → relax epsilon
      2. Positive curvature rejection → relax epsilon (avoid lock-in)
      3. Ill-conditioning (neg curvature, tiny step, kappa explosion, angle decay) → tighten epsilon
    
    Trial matrix mechanism: if H^{trial} has kappa > kappa_max, reject it.
    """
    n = len(x0)
    if H0 is None:
        H = np.eye(n)
    else:
        H = H0.copy()
    
    if eps0 is None:
        eps0 = eps_max
    
    x = x0.copy()
    g = grad_f(x)
    f_val = f(x)
    
    eps_k = eps0
    
    c_good = 0     # consecutive good descent counter
    c_posrej = 0   # consecutive positive-curvature rejection counter
    
    nfev = 1
    ngev = 1
    
    history = {
        'f_vals': [f_val],
        'gnorms': [norm(g, np.inf)],
        'eps_vals': [eps_k],
        'skip_count': 0,
        'ill_count': 0,
        'cond_nums': [],
        'alphas': [],
    }
    
    for k in range(max_iter):
        # Check convergence
        gnorm = norm(g, np.inf)
        if gnorm <= tol:
            history['message'] = 'Converged'
            break
        
        # Compute search direction
        d = -H @ g
        cos_theta = -g.dot(d) / (norm(g) * norm(d)) if norm(g) > 0 and norm(d) > 0 else 1.0
        
        # Line search
        alpha, ls_ok = wolfe_line_search(f, grad_f, x, d, g)
        if alpha < 1e-20:
            history['message'] = 'Line search failed'
            break
        
        history['alphas'].append(alpha)
        
        x_new = x + alpha * d
        s = alpha * d
        g_new = grad_f(x_new)
        y = g_new - g
        f_new = f(x_new)
        
        nfev += 1
        ngev += 1
        
        # ---- Feedback control logic ----
        flag_ill = False
        
        # Check if we should accept BFGS update
        if y.dot(s) / (norm(s)**2 + 1e-15) >= eps_k * norm(g)**1.0:
            # Try BFGS update on H
            H_trial = bfgs_update_h(H, s, y)
            
            # Trial matrix condition number check
            if estimate_condition_number(H_trial) > kappa_max:
                # Reject bad update
                H = H  # keep old H
                flag_ill = True
            else:
                H = H_trial  # accept
            
            c_posrej = 0
            
            # Additional ill-conditioning checks
            if alpha < alpha_min:
                flag_ill = True
            if cos_theta < c_min:
                flag_ill = True
        else:
            # Curvature condition not met
            H = H  # skip update
            if y.dot(s) > 0:
                # Positive curvature but rejected → over-conservative
                c_posrej += 1
            else:
                # Negative curvature → genuine ill-conditioning
                flag_ill = True
        
        # Update descent counter
        if f_new < f_val and (f_val - f_new) / max(1.0, abs(f_val)) >= tau:
            c_good += 1
        else:
            c_good = 0
        
        # ---- Adapt epsilon ----
        if flag_ill:
            eps_k = min(eps_max, eps_k * rho_up)
            c_good = 0
            c_posrej = 0
            history['ill_count'] += 1
        elif c_good >= m1 or c_posrej >= m3:
            eps_k = max(eps_min, eps_k * rho_down)
            c_good = 0
            c_posrej = 0
        # else: eps_k unchanged
        
        # Track history
        history['f_vals'].append(f_new)
        history['gnorms'].append(gnorm)
        history['eps_vals'].append(eps_k)
        if not flag_ill and norm(s) > 1e-15:
            hs = 'updated' if np.allclose(H @ s, H @ s) else 'skipped'
        else:
            history['skip_count'] += 1
        
        # Condition number (only occasionally to reduce cost)
        if k % 10 == 0 and n <= 200:
            history['cond_nums'].append(estimate_condition_number(H))
        
        # Update state
        x = x_new
        g = g_new
        f_val = f_new
        
        if callback:
            callback(k, x, f_val, gnorm, eps_k)
    
    history['iterations'] = k + 1
    history['nfev'] = nfev
    history['ngev'] = ngev
    
    return x, f_val, history


def bfgs_standard(f, grad_f, x0, max_iter=10000, tol=1e-6, H0=None, **kwargs):
    """Standard BFGS (no cautious modification)."""
    n = len(x0)
    H = np.eye(n) if H0 is None else H0.copy()
    x = x0.copy()
    g = grad_f(x)
    f_val = f(x)
    
    history = {'f_vals': [f_val], 'gnorms': [norm(g, np.inf)], 'skip_count': 0}
    
    for k in range(max_iter):
        if norm(g, np.inf) <= tol:
            history['message'] = 'Converged'
            break
        
        d = -H @ g
        alpha, ls_ok = wolfe_line_search(f, grad_f, x, d, g)
        if alpha < 1e-20:
            history['message'] = 'Line search failed'
            break
        
        x_new = x + alpha * d
        s = alpha * d
        g_new = grad_f(x_new)
        y = g_new - g
        f_new = f(x_new)
        
        if y.dot(s) > 1e-15:
            H = bfgs_update_h(H, s, y)
        else:
            history['skip_count'] += 1
        
        history['f_vals'].append(f_new)
        history['gnorms'].append(norm(g, np.inf))
        
        x = x_new
        g = g_new
        f_val = f_new
    
    history['iterations'] = k + 1
    history['nfev'] = k + 2
    history['ngev'] = k + 2
    return x, f_val, history


def cbfgs_lf(f, grad_f, x0, max_iter=10000, tol=1e-6, H0=None,
             epsilon=1e-6, **kwargs):
    """Li-Fukushima Cautious BFGS with fixed epsilon."""
    n = len(x0)
    H = np.eye(n) if H0 is None else H0.copy()
    x = x0.copy()
    g = grad_f(x)
    f_val = f(x)
    
    history = {'f_vals': [f_val], 'gnorms': [norm(g, np.inf)], 'skip_count': 0}
    
    for k in range(max_iter):
        if norm(g, np.inf) <= tol:
            history['message'] = 'Converged'
            break
        
        d = -H @ g
        alpha, ls_ok = wolfe_line_search(f, grad_f, x, d, g)
        if alpha < 1e-20:
            history['message'] = 'Line search failed'
            break
        
        x_new = x + alpha * d
        s = alpha * d
        g_new = grad_f(x_new)
        y = g_new - g
        f_new = f(x_new)
        
        # Li-Fukushima cautious condition
        if y.dot(s) / (norm(s)**2 + 1e-15) >= epsilon * norm(g):
            H = bfgs_update_h(H, s, y)
        else:
            history['skip_count'] += 1
        
        history['f_vals'].append(f_new)
        history['gnorms'].append(norm(g, np.inf))
        
        x = x_new
        g = g_new
        f_val = f_new
    
    history['iterations'] = k + 1
    history['nfev'] = k + 2
    history['ngev'] = k + 2
    return x, f_val, history


def damped_bfgs_powell(f, grad_f, x0, max_iter=10000, tol=1e-6, H0=None, **kwargs):
    """Powell's Damped BFGS (1978).
    y_k is replaced by theta*y_k + (1-theta)*B_k*s_k where theta ensures y^T s >= 0.2 s^T B s.
    """
    n = len(x0)
    # Powell's damped BFGS uses the direct Hessian approximation B
    B = np.eye(n) if H0 is None else np.linalg.inv(H0)
    x = x0.copy()
    g = grad_f(x)
    f_val = f(x)
    
    history = {'f_vals': [f_val], 'gnorms': [norm(g, np.inf)], 'skip_count': 0}
    
    for k in range(max_iter):
        if norm(g, np.inf) <= tol:
            history['message'] = 'Converged'
            break
        
        # Solve B d = -g
        try:
            d = solve(B, -g, assume_a='pos')
        except:
            d = -g  # fallback to gradient descent
        
        alpha, ls_ok = wolfe_line_search(f, grad_f, x, d, g)
        if alpha < 1e-20:
            history['message'] = 'Line search failed'
            break
        
        x_new = x + alpha * d
        s = alpha * d
        g_new = grad_f(x_new)
        y = g_new - g
        f_new = f(x_new)
        
        yTs = y.dot(s)
        sTBs = s.dot(B @ s)
        
        if yTs >= 0.2 * sTBs:
            theta = 1.0
        else:
            theta = 0.8 * sTBs / (sTBs - yTs + 1e-15)
        
        y_hat = theta * y + (1.0 - theta) * (B @ s)
        
        B = bfgs_update_b(B, s, y_hat)
        
        history['f_vals'].append(f_new)
        history['gnorms'].append(norm(g, np.inf))
        
        x = x_new
        g = g_new
        f_val = f_new
    
    history['iterations'] = k + 1
    history['nfev'] = k + 2
    history['ngev'] = k + 2
    return x, f_val, history


def lbfgs(f, grad_f, x0, max_iter=10000, tol=1e-6, m=5, H0=None, **kwargs):
    """Limited-memory BFGS (L-BFGS) with memory parameter m."""
    n = len(x0)
    x = x0.copy()
    g = grad_f(x)
    f_val = f(x)
    
    history = {'f_vals': [f_val], 'gnorms': [norm(g, np.inf)], 'skip_count': 0}
    
    # Storage for s and y vectors
    s_list = []
    y_list = []
    
    for k in range(max_iter):
        if norm(g, np.inf) <= tol:
            history['message'] = 'Converged'
            break
        
        # Two-loop recursion for L-BFGS direction
        d = _lbfgs_two_loop(g, s_list, y_list, H0)
        
        alpha, ls_ok = wolfe_line_search(f, grad_f, x, d, g)
        if alpha < 1e-20:
            history['message'] = 'Line search failed'
            break
        
        x_new = x + alpha * d
        s = alpha * d
        g_new = grad_f(x_new)
        y = g_new - g
        f_new = f(x_new)
        
        yTs = y.dot(s)
        if yTs > 1e-15:
            s_list.append(s.copy())
            y_list.append(y.copy())
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        else:
            history['skip_count'] += 1
        
        history['f_vals'].append(f_new)
        history['gnorms'].append(norm(g, np.inf))
        
        x = x_new
        g = g_new
        f_val = f_new
    
    history['iterations'] = k + 1
    history['nfev'] = k + 2
    history['ngev'] = k + 2
    return x, f_val, history


def _lbfgs_two_loop(grad, s_list, y_list, H0=None):
    """L-BFGS two-loop recursion to compute d = -H * grad."""
    if len(s_list) == 0:
        return -grad
    
    q = grad.copy()
    alpha_list = []
    
    # First loop
    for s, y in zip(reversed(s_list), reversed(y_list)):
        rho = 1.0 / y.dot(s)
        alpha = rho * s.dot(q)
        alpha_list.append(alpha)
        q = q - alpha * y
    
    # Initial approximation
    if H0 is None:
        # Scale H0 based on most recent curvature
        s_latest = s_list[-1]
        y_latest = y_list[-1]
        gamma = s_latest.dot(y_latest) / y_latest.dot(y_latest)
        r = gamma * q
    else:
        r = H0 @ q
    
    # Second loop
    for s, y, alpha in zip(s_list, y_list, reversed(alpha_list)):
        rho = 1.0 / y.dot(s)
        beta = rho * y.dot(r)
        r = r + s * (alpha - beta)
    
    return -r


def gradient_descent(f, grad_f, x0, max_iter=10000, tol=1e-6, lr=0.01,
                     momentum=0.9, **kwargs):
    """Gradient descent with momentum."""
    x = x0.copy()
    g = grad_f(x)
    f_val = f(x)
    v = np.zeros_like(x)
    
    history = {'f_vals': [f_val], 'gnorms': [norm(g, np.inf)], 'skip_count': 0}
    
    for k in range(max_iter):
        if norm(g, np.inf) <= tol:
            history['message'] = 'Converged'
            break
        
        v = momentum * v - lr * g
        x_new = x + v
        g_new = grad_f(x_new)
        f_new = f(x_new)
        
        # If function increased, reduce learning rate
        if f_new > f_val:
            lr *= 0.5
            v = np.zeros_like(x)
            x_new = x - lr * g
            g_new = grad_f(x_new)
            f_new = f(x_new)
        
        history['f_vals'].append(f_new)
        history['gnorms'].append(norm(g, np.inf))
        
        x = x_new
        g = g_new
        f_val = f_new
    
    history['iterations'] = k + 1
    history['nfev'] = k + 2
    history['ngev'] = k + 2
    return x, f_val, history


# ============================================================================
# Solver Registry
# ============================================================================

SOLVERS = {
    'AC-BFGS': acbfgs,
    'BFGS': bfgs_standard,
    'LF-6': lambda f, g, x0, **kw: cbfgs_lf(f, g, x0, epsilon=1e-6, **kw),
    'LF-4': lambda f, g, x0, **kw: cbfgs_lf(f, g, x0, epsilon=1e-4, **kw),
    'LF-2': lambda f, g, x0, **kw: cbfgs_lf(f, g, x0, epsilon=1e-2, **kw),
    'PD-BFGS': damped_bfgs_powell,
    'L-BFGS5': lambda f, g, x0, **kw: lbfgs(f, g, x0, m=5, **kw),
    'L-BFGS10': lambda f, g, x0, **kw: lbfgs(f, g, x0, m=10, **kw),
    'GD': gradient_descent,
}

# AC-BFGS ablation variants
SOLVERS_ABLATION = {
    'A0-LF6': lambda f, g, x0, **kw: cbfgs_lf(f, g, x0, epsilon=1e-6, **kw),  # baseline
    'A1-Descent': lambda f, g, x0, **kw: acbfgs(f, g, x0, rho_up=1.0, m3=9999,
                                                   kappa_max=1e30, alpha_min=0, c_min=0, **kw),
    'A2-PosRej': lambda f, g, x0, **kw: acbfgs(f, g, x0, m1=9999,
                                                  kappa_max=1e30, alpha_min=0, c_min=0,
                                                  rho_up=1.0, **kw),
    'A3-IllCond': lambda f, g, x0, **kw: acbfgs(f, g, x0, m1=9999, m3=9999, **kw),
    'A4-ACBFGS': acbfgs,
}


if __name__ == "__main__":
    # Quick sanity test
    from test_problems import ExtendedRosenbrock
    prob = ExtendedRosenbrock(10)
    print(f"Testing {prob.name} (n={prob.n})")
    
    x_sol, f_sol, hist = acbfgs(prob.eval, prob.grad, prob.x0, max_iter=200)
    print(f"  AC-BFGS: f={f_sol:.6e}, iters={hist['iterations']}, skip={hist['skip_count']}")
