# acbfgs — Adaptive Cautious BFGS for Nonconvex Optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20323687.svg)](https://doi.org/10.5281/zenodo.20323687)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Open-source Python implementation of **AC-BFGS** (Adaptive Cautious BFGS), a quasi-Newton method for nonconvex unconstrained optimization. This repository accompanies the paper:

> **Adaptive Cautious BFGS Methods for Nonconvex Unconstrained Optimization:
> Algorithm, Implementation and Reproducible Benchmarking**

## Features

- **AC-BFGS**: Feedback-controlled adaptive cautious BFGS with three complementary signals
  (descent feedback, positive-rejection feedback, ill-conditioning protection)
- **Comparison solvers**: Standard BFGS, Li--Fukushima CBFGS (ε = 10⁻², 10⁻⁴, 10⁻⁶),
  Powell Damped BFGS, L-BFGS (m = 5, 10), Gradient Descent
- **60+ test problems**: Classical CUTEst functions implemented from mathematical
  definitions, modern nonconvex problems (phase retrieval, matrix factorization,
  robust regression), and structured stress tests
- **Ablation study**: Five variants (A0--A4) isolating each feedback signal
- **Parameter sensitivity**: One-factor-at-a-time (OFAT) analysis on 8 hyperparameters
- **Reproducible**: Single command to reproduce all experiments and figures

## Installation

```bash
git clone https://github.com/MiyaTomono-new/acbfgs.git
cd acbfgs
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, NumPy, SciPy, Matplotlib

## Quick Start

```python
from acbfgs.algorithms import acbfgs
from acbfgs.test_problems import ExtendedRosenbrock

prob = ExtendedRosenbrock(100)
x_opt, f_opt, history = acbfgs(prob.eval, prob.grad, prob.x0, max_iter=5000)
print(f"f* = {f_opt:.6e}, iterations = {history['iterations']}")
```

## Reproducing Paper Results

### Full Benchmark (60 problems, 5000 max iter)

```bash
python experiments/run_benchmark.py --max-dim 200 --output results.json
```

### Ablation Study

```bash
python experiments/run_ablation.py
```

### Parameter Sensitivity (OFAT)

```bash
python experiments/run_sensitivity.py
```

### Generate All Figures

```bash
python experiments/generate_figures.py
```

## Package Structure

```
acbfgs/
├── algorithms.py          # AC-BFGS + 8 comparison solvers
├── test_problems.py       # 60+ test problem implementations
├── benchmark.py           # Automated benchmarking pipeline
├── profiles.py            # Performance & data profile generation
├── sensitivity.py         # OFAT parameter sensitivity analysis
├── __init__.py
├── experiments/
│   ├── run_benchmark.py   # Full benchmark script
│   ├── run_sensitivity.py # Parameter sensitivity script
│   └── generate_figures.py # Figure generation
├── figures/               # Output figures (PNG)
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this software in your research, please cite:

```bibtex
@article{acbfgs2025,
  title={Adaptive Cautious BFGS Methods for Nonconvex Unconstrained
         Optimization: Algorithm, Implementation and Reproducible Benchmarking},
  author={Author Name},
  journal={Optimization Methods and Software},
  year={2025},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
