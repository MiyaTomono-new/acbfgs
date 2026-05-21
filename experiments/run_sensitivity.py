"""Parameter sensitivity: One-Factor-At-a-Time (OFAT) analysis for AC-BFGS.
Runs AC-BFGS with one parameter varied while others held at defaults,
on a 20-problem CUTEst subset.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from test_problems_extended import build_extended_suite
from algorithms import acbfgs

# Default parameters
DEFAULTS = {
    'rho_down': 0.9, 'rho_up': 1.2, 'm1': 3, 'm3': 2,
    'eps_min': 1e-6, 'eps_max': 1e-2, 'tau': 1e-8,
    'alpha_min': 1e-12, 'kappa_max': 1e10, 'c_min': 1e-6,
}

# OFAT sweep values
SWEEP = {
    'rho_down': [0.7, 0.8, 0.9, 0.95],
    'rho_up': [1.1, 1.2, 1.5, 2.0],
    'm1': [1, 3, 5, 10],
    'm3': [1, 2, 5, 10],
    'eps_min': [1e-8, 1e-6, 1e-4],
    'eps_max': [1e-4, 1e-2, 1e-1],
    'alpha_min': [1e-16, 1e-12, 1e-8],
    'kappa_max': [1e8, 1e10, 1e12],
}

# Get 20-problem subset (first 20 from extended suite, all dims)
all_problems = build_extended_suite(max_dim=100)
# Take representative subset across categories
subset = all_problems[:20]  # first 20

print(f"Parameter sensitivity on {len(subset)} problems")
print(f"Sweeping {len(SWEEP)} parameters\n")

results = []

for param_name, values in SWEEP.items():
    default_val = DEFAULTS[param_name]
    print(f"{'='*60}")
    print(f"Parameter: {param_name} (default={default_val})")
    print(f"{'='*60}")
    
    for val in values:
        params = DEFAULTS.copy()
        params[param_name] = val
        params['eps0'] = params['eps_max']
        
        solve_count = 0
        total_iters = 0
        total_skip = 0
        
        for prob in subset:
            try:
                x_sol, f_sol, hist = acbfgs(
                    prob.eval, prob.grad, prob.x0.copy(),
                    max_iter=3000, tol=1e-6,
                    **{k: v for k, v in params.items() if k in DEFAULTS}
                )
                gnorm = np.linalg.norm(prob.grad(x_sol), np.inf)
                if gnorm <= 1e-6:
                    solve_count += 1
                    total_iters += hist.get('iterations', 3000)
                    total_skip += hist.get('skip_count', 0)
            except Exception as e:
                pass
        
        avg_iters = total_iters / solve_count if solve_count > 0 else 3000
        avg_skip = total_skip / (max(solve_count * avg_iters, 1))
        
        is_default = (val == default_val)
        marker = "*" if is_default else " "
        print(f"  {marker} {param_name}={str(val):10s} -> solved={solve_count}/{len(subset)} "
              f"avg_iters={avg_iters:.0f} avg_skip={avg_skip:.3f}")
        
        results.append({
            'parameter': param_name,
            'value': str(val),
            'is_default': is_default,
            'solve_count': solve_count,
            'total_problems': len(subset),
            'avg_iterations': avg_iters,
            'avg_skip_ratio': avg_skip,
        })

# Save
output = {'results': results, 'defaults': {k: str(v) for k,v in DEFAULTS.items()}}

def convert(o):
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, dict): return {str(k): convert(v) for k,v in o.items()}
    if isinstance(o, list): return [convert(v) for v in o]
    return str(o)

with open("sensitivity_results.json", "w") as f:
    json.dump(convert(output), f, indent=2)

# Print summary
print(f"\n{'='*60}")
print("SENSITIVITY SUMMARY: Solve rate deviation from default")
print(f"{'='*60}")
for param_name in SWEEP:
    pr = [r for r in results if r['parameter'] == param_name]
    default_r = next((r for r in pr if r['is_default']), None)
    if default_r:
        base = default_r['solve_count']
        print(f"\n{param_name}:")
        for r in pr:
            delta = r['solve_count'] - base
            sym = ">" if delta > 0 else ("<" if delta < 0 else "=")
            print(f"  {r['value']:10s} solved={r['solve_count']}/20 ({sym}{abs(delta)}) "
                  f"iters={r['avg_iterations']:.0f}")
