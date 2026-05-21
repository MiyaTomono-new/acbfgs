"""Final benchmark v2: 80+ instances, 2000 max_iter, incremental saving."""
import sys, os, json, time, traceback
import numpy as np
from numpy.linalg import norm
sys.path.insert(0, os.path.dirname(__file__))

from test_problems_batch3 import build_full_suite
from algorithms import SOLVERS, acbfgs, bfgs_standard, cbfgs_lf, damped_bfgs_powell, lbfgs, gradient_descent

# Remap solvers for this script
SOLVER_MAP = {
    'AC-BFGS': acbfgs,
    'BFGS': bfgs_standard,
    'LF-6': lambda f,g,x0,**kw: cbfgs_lf(f,g,x0,epsilon=1e-6,**kw),
    'LF-4': lambda f,g,x0,**kw: cbfgs_lf(f,g,x0,epsilon=1e-4,**kw),
    'LF-2': lambda f,g,x0,**kw: cbfgs_lf(f,g,x0,epsilon=1e-2,**kw),
    'PD-BFGS': damped_bfgs_powell,
    'L-BFGS5': lambda f,g,x0,**kw: lbfgs(f,g,x0,m=5,**kw),
    'L-BFGS10': lambda f,g,x0,**kw: lbfgs(f,g,x0,m=10,**kw),
    'GD': gradient_descent,
}

def eval_grad_safe(prob, x):
    try:
        g = prob.grad(x)
        if not np.all(np.isfinite(g)):
            g[~np.isfinite(g)] = 0.0
        return g
    except:
        return np.zeros(prob.n)

ps = build_full_suite(200)[:60]  # 60 instances, 5000 iter each

results = []
output_file = "results_final.json"

for pi, prob in enumerate(ps):
    print(f"[{pi+1}/85] {prob.name:30s} n={prob.n:4d}")
    for s_name, s_fn in SOLVER_MAP.items():
        if prob.n > 500 and s_name in ['BFGS','LF-6','LF-4','LF-2','AC-BFGS','PD-BFGS']:
            continue
        try:
            t0 = time.time()
            x_sol, f_sol, hist = s_fn(prob.eval, prob.grad, prob.x0.copy(), max_iter=5000, tol=1e-6)
            t_elapsed = time.time() - t0
            g_final = norm(eval_grad_safe(prob, x_sol), np.inf)
            conv = g_final <= 1e-6
            total_upd = max(hist.get('iterations', 1) - 1, 1)
            skip_ratio = hist.get('skip_count', 0) / total_upd if total_upd > 0 else 0
            
            r = {
                'problem': prob.name,
                'n': prob.n,
                'solver': s_name,
                'converged': conv,
                'f_final': float(f_sol),
                'gnorm_final': float(g_final),
                'iterations': hist.get('iterations', 2000),
                'skip_ratio': float(min(skip_ratio, 1.0)),
                'time': float(t_elapsed),
            }
            results.append(r)
            sym = "+" if conv else "-"
            print(f"  {sym} {s_name:12s} f={r['f_final']:.2e} g={r['gnorm_final']:.2e} "
                  f"iters={r['iterations']:5d} skip={r['skip_ratio']:.2f} t={t_elapsed:.2f}s")
        except Exception as e:
            print(f"  X {s_name:12s} ERROR: {str(e)[:80]}")
            results.append({
                'problem': prob.name, 'n': prob.n, 'solver': s_name,
                'converged': False, 'f_final': 1e30, 'gnorm_final': 1e10,
                'iterations': 0, 'skip_ratio': 1.0, 'time': 0.0,
            })
    
    # Incremental save every 5 problems
    if (pi + 1) % 5 == 0:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except:
            pass

# Final save
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Summary
print(f"\n{'='*70}")
print(f"FINAL SUMMARY: {len(ps)} instances")
print(f"{'='*70}")
solvers = sorted(set(r['solver'] for r in results))
print(f"{'Solver':15s} {'Solved':>8s} {'Conv%':>8s} {'Avg Iters':>10s} {'Avg Skip':>10s} {'Avg Time':>10s}")
print("-" * 65)
for s in solvers:
    sr = [r for r in results if r['solver'] == s]
    ns = sum(1 for r in sr if r['converged'])
    nt = len(sr)
    ai = np.mean([r['iterations'] for r in sr if r['converged']]) if ns else 0
    ask = np.mean([r['skip_ratio'] for r in sr if r['converged']]) if ns else 0
    at = np.mean([r['time'] for r in sr if r['converged']]) if ns else 0
    print(f"{s:15s} {ns:4d}/{nt:<4d} {100*ns/nt:6.1f}% {ai:10.1f} {ask:10.3f} {at:10.3f}s")
