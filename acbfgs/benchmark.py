"""
acbfgs - Benchmark Pipeline
=============================
Runs all solvers on all test problems, collects results, generates figures.
"""
import numpy as np
from numpy.linalg import norm
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from test_problems import build_problem_suite
from test_problems_extended import build_extended_suite
from test_problems_batch3 import build_full_suite
from algorithms import SOLVERS, SOLVERS_ABLATION

# ============================================================================
# Benchmark Runner
# ============================================================================

def run_single(problem, solver_name, solver_fn, max_iter=2000, tol=1e-6, **kwargs):
    """Run a single solver on a single problem. Returns result dict."""
    x0 = problem.x0.copy()
    t_start = time.time()
    try:
        x_sol, f_sol, hist = solver_fn(problem.eval, problem.grad, x0,
                                         max_iter=max_iter, tol=tol, **kwargs)
        t_elapsed = time.time() - t_start
        
        # Compute final gradient for reliable convergence check
        g_sol = eval_grad_safe(problem, x_sol)
        gnorm_final = float(norm(g_sol, np.inf))
        
    except Exception as e:
        return {
            'problem': problem.name,
            'n': problem.n,
            'solver': solver_name,
            'status': 'error',
            'error': str(e),
            'time': 0.0,
            'f_final': 1e10,
            'gnorm_final': 1e10,
            'iterations': 0,
            'skip_ratio': 1.0,
            'converged': False,
        }
    
    iter_count = hist.get('iterations', max_iter)
    total_updates = max(iter_count - 1, 1)
    skip_ratio = hist.get('skip_count', 0) / total_updates if total_updates > 0 else 0
    
    return {
        'problem': problem.name,
        'n': problem.n,
        'solver': solver_name,
        'status': hist.get('message', 'max_iter'),
        'time': t_elapsed,
        'f_final': float(f_sol),
        'gnorm_final': float(gnorm_final),
        'iterations': iter_count,
        'skip_ratio': float(min(skip_ratio, 1.0)),
        'converged': gnorm_final <= tol,
    }


def eval_grad_safe(problem, x):
    """Safely evaluate gradient, return large vector on error."""
    try:
        return problem.grad(x)
    except Exception:
        return np.ones(problem.n) * 1e10


def run_benchmark(max_problems=None, max_dim=200, fast=False):
    """Run full benchmark: all solvers on all problems."""
    
    if fast:
        problems = build_full_suite(max_dim=50)[:20]
        max_iter = 500
    else:
        problems = build_full_suite(max_dim=max_dim)
        max_iter = 5000
    
    if max_problems:
        problems = problems[:max_problems]
    
    print(f"Running benchmark on {len(problems)} problems with {len(SOLVERS)} solvers")
    print(f"  max_iter={max_iter}, tol=1e-6")
    print()
    
    results = []
    
    for i, prob in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {prob.name} (n={prob.n})")
        
        for solver_name, solver_fn in SOLVERS.items():
            # Skip full-memory solvers for large problems
            if prob.n > 500 and solver_name in ['BFGS', 'LF-6', 'LF-4', 'LF-2', 'AC-BFGS', 'PD-BFGS']:
                continue
            
            result = run_single(prob, solver_name, solver_fn, max_iter=max_iter)
            status_symbol = '+' if result['converged'] else '-'
            print(f"  {status_symbol} {solver_name:12s} f={result['f_final']:.3e} "
                  f"gnorm={result['gnorm_final']:.2e} iters={result['iterations']:5d} "
                  f"skip={result['skip_ratio']:.2f} time={result['time']:.2f}s")
            results.append(result)
        
        # Also run ablation variants on first 20 problems
        if i < 20 and prob.n <= 200:
            for solver_name, solver_fn in SOLVERS_ABLATION.items():
                if prob.n > 500 and solver_name in ['A0-LF6', 'A1-Descent', 'A2-PosRej', 'A3-IllCond', 'A4-ACBFGS']:
                    continue
                result = run_single(prob, solver_name, solver_fn, max_iter=max_iter)
                status_symbol = '+' if result['converged'] else '-'
                print(f"  {status_symbol} {solver_name:12s} f={result['f_final']:.3e} "
                      f"gnorm={result['gnorm_final']:.2e} iters={result['iterations']:5d} "
                      f"skip={result['skip_ratio']:.2f} time={result['time']:.2f}s")
                results.append(result)
        
        print()
    
    return results


def summarize(results):
    """Print summary statistics."""
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    solvers = sorted(set(r['solver'] for r in results))
    problems = sorted(set(r['problem'] for r in results))
    
    print(f"\n{'Solver':15s} {'Solved':>8s} {'Conv%':>8s} {'Avg Iters':>10s} "
          f"{'Avg Skip':>10s} {'Avg Time':>10s}")
    print("-" * 65)
    
    for s_name in solvers:
        s_results = [r for r in results if r['solver'] == s_name]
        n_total = len(s_results)
        n_solved = sum(1 for r in s_results if r['converged'])
        avg_iters = np.mean([r['iterations'] for r in s_results if r['converged']]) if n_solved > 0 else 0
        avg_skip = np.mean([r['skip_ratio'] for r in s_results if r['converged']]) if n_solved > 0 else 0
        avg_time = np.mean([r['time'] for r in s_results if r['converged']]) if n_solved > 0 else 0
        
        print(f"{s_name:15s} {n_solved:4d}/{n_total:<4d} {100*n_solved/n_total:6.1f}% "
              f"{avg_iters:10.1f} {avg_skip:10.3f} {avg_time:10.3f}s")
    
    return solvers, problems


def generate_performance_profiles(results, solvers, problems):
    """Generate performance profile data (Dolan-More style)."""
    # For each problem, find the best solver (minimum iterations among converged)
    profiles = {}
    
    for s_name in solvers:
        s_results = [r for r in results if r['solver'] == s_name]
        if not s_results:
            continue
        
        tau_values = np.logspace(-1, -7, 7)
        profile = {}
        
        for tau in tau_values:
            count = 0
            for prob_name in problems:
                prob_results = [r for r in s_results if r['problem'] == prob_name]
                if not prob_results:
                    continue
                r = prob_results[0]
                if r['converged'] and r['gnorm_final'] <= tau:
                    count += 1
            
            profile[float(tau)] = count / len(problems) if problems else 0
        
        profiles[s_name] = profile
    
    return profiles


def generate_data_profiles(results, solvers, problems):
    """Generate data profile data (More-Wild style)."""
    profiles = {}
    budgets = np.logspace(1, 4, 20)  # 10 to 10000 function evals
    
    for s_name in solvers:
        s_results = [r for r in results if r['solver'] == s_name]
        if not s_results:
            continue
        
        profile = {}
        for budget in budgets:
            count = 0
            for prob_name in problems:
                prob_results = [r for r in s_results if r['problem'] == prob_name]
                if not prob_results:
                    continue
                r = prob_results[0]
                if r['converged'] and r['iterations'] <= budget:
                    count += 1
            profile[float(budget)] = count / len(problems) if problems else 0
        
        profiles[s_name] = profile
    
    return profiles


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AC-BFGS benchmark')
    parser.add_argument('--fast', action='store_true', help='Fast mode (fewer problems, fewer iters)')
    parser.add_argument('--max-problems', type=int, default=None)
    parser.add_argument('--max-dim', type=int, default=200)
    parser.add_argument('--output', type=str, default='results.json')
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(max_problems=args.max_problems, max_dim=args.max_dim, fast=args.fast)
    
    # Summarize
    solvers, problems = summarize(results)
    
    # Generate profiles
    perf_profiles = generate_performance_profiles(results, solvers, problems)
    data_profiles = generate_data_profiles(results, solvers, problems)
    
    # Save results
    output_data = {
        'results': results,
        'performance_profiles': perf_profiles,
        'data_profiles': data_profiles,
        'solvers': list(solvers),
        'problems': list(problems),
    }
    
    # Convert numpy types for JSON
    def convert(o):
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return str(o)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(convert(output_data), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {args.output}")
    
    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Performance profile plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Performance profiles
        ax = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(solvers)))
        for idx, s_name in enumerate(solvers):
            if s_name in perf_profiles:
                taus = sorted(perf_profiles[s_name].keys())
                vals = [perf_profiles[s_name][t] for t in taus]
                ax.semilogx(taus, vals, 'o-', color=colors[idx], label=s_name, linewidth=2)
        
        ax.set_xlabel('Precision $\\tau$', fontsize=12)
        ax.set_ylabel('Fraction of problems solved', fontsize=12)
        ax.set_title('Performance Profiles', fontsize=14)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1e-8, 1)
        ax.set_ylim(0, 1.05)
        
        # Right: Data profiles
        ax = axes[1]
        for idx, s_name in enumerate(solvers):
            if s_name in data_profiles:
                budgets = sorted(data_profiles[s_name].keys())
                vals = [data_profiles[s_name][b] for b in budgets]
                ax.semilogx(budgets, vals, 's-', color=colors[idx], label=s_name, linewidth=2)
        
        ax.set_xlabel('Function evaluation budget', fontsize=12)
        ax.set_ylabel('Fraction of problems solved', fontsize=12)
        ax.set_title('Data Profiles', fontsize=14)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(10, 10000)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('profiles.png', dpi=150, bbox_inches='tight')
        print("Saved profiles.png")
        
        # Skip ratio comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        skip_data = {}
        for s_name in solvers:
            s_skips = [r['skip_ratio'] for r in results if r['solver'] == s_name and r['converged']]
            if s_skips:
                skip_data[s_name] = s_skips
        
        if skip_data:
            positions = range(len(skip_data))
            bp = ax.boxplot(skip_data.values(), positions=positions, widths=0.6)
            ax.set_xticklabels(skip_data.keys(), rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Skip Ratio', fontsize=12)
            ax.set_title('BFGS Update Skip Ratio Distribution', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('skip_ratios.png', dpi=150, bbox_inches='tight')
            print("Saved skip_ratios.png")
        
        # Convergence rate comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        conv_data = {}
        for s_name in solvers:
            s_iters = [r['iterations'] for r in results if r['solver'] == s_name and r['converged']]
            if s_iters:
                conv_data[s_name] = s_iters
        
        if conv_data:
            positions = range(len(conv_data))
            ax.boxplot(conv_data.values(), positions=positions, widths=0.6)
            ax.set_xticklabels(conv_data.keys(), rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Iterations to Convergence', fontsize=12)
            ax.set_title('Iteration Count Distribution', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('iterations.png', dpi=150, bbox_inches='tight')
            print("Saved iterations.png")
        
        plt.close('all')
    except ImportError:
        print("matplotlib not available, skipping plots")
    except Exception as e:
        print(f"Plot error: {e}")
