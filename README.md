# CL677: Modelling Stochastic and Turbulent Transport

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Marimo](https://img.shields.io/badge/Marimo-0.19.4-purple)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**Course:** CL 677 - Modelling Stochastic and Turbulent Transport (Spring 2025-26)
**Instructor:** Prof. Jason Picardo
**Students:** Sarthak Mishra (22b0432) | Pratyush Ranjan (22b0326)

---

## 📚 Course Description

This repository contains assignments and projects for CL677, focusing on stochastic processes, turbulent transport phenomena, and computational modeling techniques in fluid dynamics.

### Topics Covered
- Random walks and Brownian motion
- Stochastic differential equations (SDEs)
- Turbulent transport and diffusion
- Numerical methods for stochastic processes
- Monte Carlo simulations

---

## 📁 Repository Structure

```
CL677/
├── .venv/                        # Python virtual environment
├── assignments/                  # Course assignments
│   ├── Assignment1/             # Random Walks Simulation
│   │   ├── notebook.py           # Interactive Marimo notebook
│   │   └── README.md             # Assignment-specific documentation
│   └── Assignment2/              # (Coming soon)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (recommended: Python 3.13)
- **uv** (fast Python package installer) or **pip**
- **Git** (for cloning the repository)

### Installation

#### Option 1: Using `uv` (Recommended - Fast)

```bash
# Clone the repository
git clone https://github.com/sithtsar/CL677.git
cd CL677

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install marimo numpy matplotlib

# Verify installation
marimo --version
```

#### Option 2: Using `pip` (Traditional)

```bash
# Clone the repository
git clone https://github.com/sithtsar/CL677.git
cd CL677

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install marimo numpy matplotlib

# Verify installation
marimo --version
```

---

## 📓 Working with Marimo Notebooks

### What is Marimo?

[Marimo](https://marimo.io) is a reactive Python notebook that combines the best of Jupyter notebooks with reproducibility and reactivity. Key features:

- ✅ **Reactive execution** - cells automatically update when dependencies change
- ✅ **No hidden state** - guaranteed reproducibility
- ✅ **Git-friendly** - notebooks are pure Python files
- ✅ **Interactive widgets** - sliders, dropdowns, and more
- ✅ **Beautiful output** - professional plots and formatting

### Running Notebooks

#### Interactive Mode (with UI)
```bash
cd assignments/Assignment1
source ../../.venv/bin/activate
marimo edit notebook.py
```

This opens the notebook in your browser with:
- Interactive sliders for parameters
- Real-time plot updates
- Full editing capabilities

#### App Mode (read-only)
```bash
marimo run notebook.py
```

This runs the notebook as a read-only web app.

#### Script Mode (command-line)
```bash
python notebook.py
```

This executes the notebook and generates all outputs.

### Exporting to PDF

```bash
# Method 1: Export from browser
# Open notebook in edit mode, then use browser's Print → Save as PDF

# Method 2: Using marimo export (HTML)
marimo export html notebook.py -o notebook.html
# Then open in browser and print to PDF

# Method 3: Direct to HTML for conversion
marimo export html notebook_clean.py > output.html
```

---

## 📋 Assignments

### Assignment 1: Random Walks Simulation

**Status:** ✅ Completed
**Directory:** `assignments/Assignment1/`
**Topics:** Brownian motion, Euler-Maruyama method, MSD analysis

**Key Questions:**
1. Calculate mean-squared displacement and verify diffusive behavior (α = 1)
2. Convergence analysis with different time steps
3. Compare Gaussian vs. uniform random number distributions

**Quick Run:**
```bash
cd assignments/Assignment1
source ../../.venv/bin/activate
marimo edit notebook_clean.py  # For submission version
# OR
marimo edit notebook.py         # For interactive exploration
```

See [assignments/Assignment1/README.md](assignments/Assignment1/README.md) for detailed documentation.

---

## 🛠️ Development Setup

### Recommended Tools

- **Editor:** VS Code with Python and Marimo extensions
- **Python:** Python 3.13+ with `uv` for fast package management
- **Version Control:** Git with `.gitignore` for Python projects

### Project Dependencies

```txt
marimo>=0.19.4      # Reactive notebook framework
numpy>=1.24.0       # Numerical computing
matplotlib>=3.7.0   # Plotting and visualization
```

### Installing Additional Packages

```bash
source .venv/bin/activate

# Using uv (fast)
uv pip install package-name

# Using pip (traditional)
pip install package-name
```

---

## 📊 Key Concepts

### Stochastic Differential Equations

This course explores SDEs of the form:

$$
dx = f(x,t)dt + g(x,t)dW_t
$$

where $W_t$ is a Wiener process (Brownian motion).

### Euler-Maruyama Method

Discretization scheme for SDEs:

$$
x_{n+1} = x_n + f(x_n, t_n)\Delta t + g(x_n, t_n)\sqrt{\Delta t}\,\zeta_n
$$

where $\zeta_n \sim \mathcal{N}(0,1)$.

### Mean-Squared Displacement

For diffusive processes:

$$
\langle x^2(t) \rangle \sim t^\alpha
$$

- $\alpha = 1$: Normal diffusion (Brownian motion)
- $\alpha > 1$: Super-diffusion
- $\alpha < 1$: Sub-diffusion

---

## 🤝 Collaboration Guidelines

### Group Work
- Active collaboration within groups is **encouraged**
- Each member must understand all submitted work
- Individual viva assessments will scale group marks

### Academic Integrity
- Copying between groups is **strictly prohibited**
- Both lending and copying groups will receive **zero marks**
- Do not share code or answers across groups

---

## 📝 Submission Guidelines

### What to Submit
- One PDF per group with detailed answers
- All code must be reproducible
- Figures must be high-quality (300 DPI)
- Clear explanations and reasoning required

### Evaluation Criteria
1. **Correctness** (15 points): Accurate results and analysis
2. **Presentation** (5 points): Clear plots, comprehensive explanations
3. **Viva** (individual): Each member must explain all work

---

## 🐛 Troubleshooting

### Common Issues

**1. "marimo: command not found"**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Verify marimo is installed
pip list | grep marimo
```

**2. "ModuleNotFoundError: No module named 'marimo'"**
```bash
# Install marimo
uv pip install marimo
# OR
pip install marimo
```

**3. Plots not displaying**
```bash
# Ensure matplotlib is installed
uv pip install matplotlib
```

**4. Notebook has syntax errors**
```bash
# Validate Python syntax
python -c "import ast; ast.parse(open('notebook.py').read()); print('✓ Valid')"
```

**5. Git issues with large files**
```bash
# Check .gitignore includes
.venv/
__pycache__/
*.pyc
.DS_Store
```

---

## 📚 Resources

### Marimo Documentation
- Official Site: https://marimo.io
- Documentation: https://docs.marimo.io
- GitHub: https://github.com/marimo-team/marimo
- Tutorial: `marimo tutorial intro`

### Course References
- **Stochastic Processes:** "Stochastic Differential Equations" by Bernt Øksendal
- **Turbulence:** "Turbulent Flows" by Stephen Pope
- **Numerical Methods:** "Numerical Solution of Stochastic Differential Equations" by Kloeden & Platen

### Python & Scientific Computing
- NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/stable/
- SciPy: https://docs.scipy.org/

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Authors

- **Sarthak Mishra** (22b0432) - [GitHub](https://github.com/sithtsar)
- **Pratyush Ranjan** (22b0326)

---

## 🔗 Links

- **Course Repository:** https://github.com/sithtsar/CL677
- **Assignment 1:** [assignments/Assignment1/README.md](assignments/Assignment1/README.md)
- **Marimo:** https://marimo.io

---

**Last Updated:** January 2026
