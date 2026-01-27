# Assignment 1: Random Walks Simulation

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Marimo](https://img.shields.io/badge/Marimo-0.19.4-purple)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

**Course:** CL 677 - Modelling Stochastic and Turbulent Transport (Spring 2025-26)
**Due Date:** [Check course portal]
**Points:** 20 + 5 = 25 total

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Quick Start](#-quick-start)
- [File Descriptions](#-file-descriptions)
- [Running the Notebooks](#-running-the-notebooks)
- [Assignment Questions](#-assignment-questions)
- [Results Summary](#-results-summary)
- [Implementation Details](#-implementation-details)
- [Troubleshooting](#-troubleshooting)
- [Submission Checklist](#-submission-checklist)

---

## 🎯 Overview

This assignment explores **Brownian motion** through Monte Carlo simulations using the **Euler-Maruyama discretization** of stochastic differential equations. We simulate random walk trajectories, analyze their statistical properties, and investigate numerical convergence.

### Learning Objectives

✅ Understand Brownian motion and random walks
✅ Implement the Euler-Maruyama scheme for SDEs
✅ Analyze mean-squared displacement (MSD)
✅ Verify diffusive behavior (α = 1)
✅ Study numerical convergence with time step
✅ Apply the Central Limit Theorem

---

## 📝 Problem Statement

### The Stochastic Differential Equation

Compute trajectories of a random walk using:

$$
dx = A\zeta\sqrt{dt}
$$

**Parameters:**
- `A` = 1.0 (amplitude of Brownian force)
- `dt` = 0.02 (time step)
- `T` = 10.0 (total simulation time)
- `N_traj` ≥ 1000 (number of trajectories)
- Initial condition: `x(0) = 0`

**Random Number Distribution:**
- ζ ~ N(0, 1) - Unit-variance Gaussian distribution
- Or ζ ~ U(-√3, √3) - Unit-variance uniform distribution

### Three Questions to Answer

**Question 1 (10 points):** Calculate MSD ⟨x²⟩ vs. time, plot on log-log scale, determine slope α. Is α ≈ 1?

**Question 2 (5 points):** Test convergence with dt = [0.02, 0.005, 0.001]. Do MSD curves overlap?

**Question 3 (5 points):** Compare Gaussian vs. uniform random numbers. Same results?

**Writing & Plotting (5 points):** Quality of explanations, plots, and reasoning.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
Python 3.12+
marimo 0.19.4+
numpy 1.24+
matplotlib 3.7+
```

### Installation

```bash
# Navigate to Assignment1 directory
cd Assignment1

# Activate virtual environment
source ../.venv/bin/activate

# Install dependencies (if not already installed)
uv pip install marimo numpy matplotlib
# OR using pip
pip install marimo numpy matplotlib
```

### Run the Notebook (Interactive)

```bash
# Option 1: Clean submission version (recommended for final PDF)
marimo edit notebook_clean.py

# Option 2: Interactive version with sliders (for exploration)
marimo edit notebook.py
```

The notebook will open in your browser at `http://localhost:2718`

---

## 📁 File Descriptions

### Main Files

| File | Description | Use Case |
|------|-------------|----------|
| `notebook_clean.py` | Clean version with fixed parameters | **Final submission PDF** |
| `notebook.py` | Interactive version with sliders | Exploration and parameter tuning |

### Key Differences

**notebook_clean.py** (Recommended for Submission)
- ✅ Fixed parameters (A=1, T=10, dt=0.02, N=2000)
- ✅ Professional formatting and layout
- ✅ Comprehensive analysis and explanations
- ✅ High-quality figures (300 DPI)
- ✅ Structured exactly per assignment questions
- ✅ Ready for PDF export

**notebook.py** (Interactive Exploration)
- ✅ Interactive sliders for parameters
- ✅ Real-time updates when changing values
- ✅ Great for understanding parameter effects
- ✅ Useful for debugging and testing

### Generated Files

```
Assignment1/
├── notebook.py              # Interactive version
├── notebook_clean.py        # Submission version
├── README.md                # This file
└── (output files)           # Generated when you export
    ├── assignment1.pdf      # Exported PDF for submission
    └── figures/             # Individual figure exports
```

---

## 💻 Running the Notebooks

### Method 1: Interactive Mode (Recommended)

```bash
# Start the notebook server
marimo edit notebook_clean.py
```

**Features:**
- Interactive web interface
- Live code execution
- Real-time plot updates
- Cell-by-cell execution

**Browser opens at:** `http://localhost:2718`

### Method 2: App Mode (Read-Only)

```bash
# Run as a web application
marimo run notebook_clean.py
```

**Use case:** Share results without allowing code editing

### Method 3: Command-Line Execution

```bash
# Execute all cells and generate outputs
python notebook_clean.py
```

**Use case:** Batch processing or automation

### Method 4: Jupyter-Style (Cell-by-Cell)

```bash
# Open in interactive mode
marimo edit notebook_clean.py

# In the browser interface:
# - Click "Run" to execute individual cells
# - Use Ctrl+Enter to run current cell
# - Use Shift+Enter to run and move to next
```

---

## 📊 Assignment Questions

### Question 1: Mean-Squared Displacement (10 points)

**Objective:** Calculate MSD and verify diffusive behavior

**What the code does:**
1. Simulates 2000 trajectories with dt=0.02
2. Calculates ⟨x²⟩ at each time point
3. Plots MSD vs. time on log-log scale
4. Fits slope α from log(MSD) vs. log(t)
5. Compares with theory (α = 1)

**Expected Result:**
```
α ≈ 1.00 (within ~5% of theory)
```

**Key Code Section:**
```python
# Simulate trajectories
x_traj_q1, time_q1 = simulate_trajectories(A=1.0, T=10.0, dt=0.02, N_traj=2000)

# Calculate MSD
msd_q1 = calculate_msd(x_traj_q1)

# Fit exponent
alpha_q1, coeffs_q1 = fit_alpha(time_q1, msd_q1)
```

**Output Plots:**
- Figure: Sample trajectories (first 50 of 2000)
- Figure: MSD vs. time (log-log) with fitted line

---

### Question 2: Convergence Analysis (5 points)

**Objective:** Verify numerical convergence with different time steps

**What the code does:**
1. Runs simulations with dt = [0.02, 0.005, 0.001]
2. Calculates MSD for each case
3. Overlays all three curves on one plot
4. Compares α values

**Expected Result:**
```
dt = 0.02:   α ≈ 1.00, N_steps = 500
dt = 0.005:  α ≈ 1.00, N_steps = 2000
dt = 0.001:  α ≈ 1.00, N_steps = 10000

Curves should overlap closely (Δα < 0.1)
```

**Key Code Section:**
```python
dt_values = [0.02, 0.005, 0.001]
for dt in dt_values:
    x_traj, time = simulate_trajectories(A=1.0, T=10.0, dt=dt, N_traj=2000)
    msd = calculate_msd(x_traj)
    alpha, _ = fit_alpha(time, msd)
```

**Output Plot:**
- Figure: Three MSD curves overlaid with different colors/markers

---

### Question 3: Distribution Comparison (5 points)

**Objective:** Compare Gaussian vs. uniform random numbers

**What the code does:**
1. Simulates with ζ ~ N(0,1) (Gaussian)
2. Simulates with ζ ~ U(-√3, √3) (uniform, unit variance)
3. Calculates MSD for both
4. Overlays and compares results

**Expected Result:**
```
α_gaussian ≈ α_uniform (within ~5%)
MSD curves should nearly overlap
Central Limit Theorem validated!
```

**Key Code Section:**
```python
# Gaussian
x_gauss, time = simulate_trajectories(A=1.0, T=10.0, dt=0.02, N_traj=2000,
                                       distribution='gaussian')
msd_gauss = calculate_msd(x_gauss)

# Uniform (unit variance: [-√3, √3])
x_unif, _ = simulate_trajectories(A=1.0, T=10.0, dt=0.02, N_traj=2000,
                                  distribution='uniform')
msd_unif = calculate_msd(x_unif)
```

**Output Plot:**
- Figure: Gaussian vs. Uniform MSD comparison

---

## 📈 Results Summary

### Our Findings

| Question | Metric | Expected | Obtained | Status |
|----------|--------|----------|----------|--------|
| Q1 | α (dt=0.02) | 1.000 | ~1.00 | ✅ |
| Q2 | Δα (convergence) | <0.1 | ~0.05 | ✅ |
| Q3 | α_gauss vs α_unif | Same | ~99% match | ✅ |

### Key Insights

1. **Diffusive Behavior Confirmed**
   - MSD ∝ t (linear on log-log plot)
   - α ≈ 1.00 validates Brownian motion theory

2. **Numerical Convergence Achieved**
   - Results converge as dt → 0
   - dt = 0.02 is sufficiently accurate
   - Euler-Maruyama scheme works well

3. **Central Limit Theorem Validated**
   - Gaussian and uniform give same MSD
   - Only variance matters, not distribution shape
   - Universality of diffusive behavior

---

## 🔧 Implementation Details

### Core Functions

#### `simulate_trajectories(A, T, dt, N_traj, distribution='gaussian')`

Simulates Brownian motion trajectories using Euler-Maruyama discretization.

**Algorithm:**
```python
1. Generate random numbers ζ ~ N(0,1) or U(-√3,√3)
2. Compute increments: dx = A·ζ·√dt
3. Cumulative sum: x(t) = Σ dx
4. Add initial condition x(0) = 0
5. Return (x_traj, time)
```

**Performance:** Vectorized operations using NumPy for speed

---

#### `calculate_msd(x_traj)`

Calculates mean-squared displacement across all trajectories.

**Formula:**
```python
MSD(t) = ⟨x²(t)⟩ = (1/N) Σᵢ xᵢ²(t)
```

**Implementation:**
```python
return np.mean(x_traj**2, axis=0)
```

---

#### `fit_alpha(time, msd)`

Fits power-law exponent from log-log data.

**Theory:**
```
MSD ~ t^α
⟹ log(MSD) = α·log(t) + const
```

**Implementation:**
```python
log_time = np.log(time[1:])  # Exclude t=0
log_msd = np.log(msd[1:])
coeffs = np.polyfit(log_time, log_msd, 1)
alpha = coeffs[0]  # Slope = exponent
```

---

### Parameter Choices

| Parameter | Value | Justification |
|-----------|-------|---------------|
| A | 1.0 | Standard amplitude for Brownian motion |
| T | 10.0 | Long enough for asymptotic behavior |
| dt | 0.02 | Balances accuracy and computation time |
| N_traj | 2000 | Ensures good statistics (>1000 minimum) |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Notebook won't start

**Error:** `marimo: command not found`

**Solution:**
```bash
# Activate virtual environment first!
source ../.venv/bin/activate
marimo --version  # Should show version number
```

---

#### 2. Module import errors

**Error:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
source ../.venv/bin/activate
uv pip install numpy matplotlib marimo
```

---

#### 3. Plots not displaying

**Error:** Figures show as blank or don't appear

**Solution:**
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# If issues persist, try:
pip install --upgrade matplotlib
```

---

#### 4. Performance issues (slow execution)

**Problem:** Simulation takes too long

**Solutions:**
```python
# Option 1: Reduce trajectories for testing
N_traj = 1000  # Instead of 2000

# Option 2: Increase time step for testing
dt = 0.05  # Instead of 0.02

# Option 3: Reduce total time
T = 5.0  # Instead of 10.0

# Note: Use original parameters for final submission!
```

---

#### 5. PDF export issues

**Problem:** Can't export to PDF directly

**Solution:**
```bash
# Method 1: Browser print
# Open notebook, use browser's Print → Save as PDF

# Method 2: Export to HTML first
marimo export html notebook_clean.py > output.html
# Open output.html in browser
# Print to PDF with Chrome/Firefox

# Tip: Use "Save as PDF" option, set margins to "None"
```

---

#### 6. Git issues with notebook

**Problem:** Git shows notebook as always modified

**Solution:**
```bash
# Marimo notebooks are pure Python - no special handling needed
# If you see unexpected changes, check:
git diff notebook.py

# Common causes:
# - Automatic line number updates
# - Cell execution order changes
# - Output caching

# To reset:
git checkout -- notebook.py
```

---

## ✅ Submission Checklist

### Before Submitting

- [ ] All three questions answered completely
- [ ] All plots are high-quality (300 DPI)
- [ ] Figures have clear labels, legends, and titles
- [ ] α values calculated and reported
- [ ] Deviations from theory explained
- [ ] Convergence analysis complete
- [ ] Distribution comparison with interpretation
- [ ] Physical explanations provided
- [ ] Code is clean and well-commented
- [ ] All cells execute without errors
- [ ] PDF exported successfully

### Running Final Checks

```bash
# 1. Activate environment
source ../.venv/bin/activate

# 2. Test notebook execution
python notebook_clean.py  # Should complete without errors

# 3. Check all plots generated
marimo run notebook_clean.py  # Visual inspection

# 4. Export to PDF
# Use browser: marimo edit notebook_clean.py → Print → Save as PDF

# 5. Verify PDF quality
# Open PDF and check:
# - All figures visible
# - Text is readable
# - Equations render correctly
# - No truncated content
```

### What to Submit

1. **PDF file** with:
   - Cover page (names, roll numbers)
   - All three questions answered
   - High-quality plots
   - Detailed explanations
   - Code snippets (optional but recommended)

2. **Group composition:**
   - One submission per group
   - All member names and roll numbers

3. **File naming:**
   ```
   CL677_Assignment1_Group[X]_RollNumbers.pdf
   Example: CL677_Assignment1_Group5_22b0432_22b0326.pdf
   ```

### Viva Preparation

Each member should be able to:
- ✅ Explain the Euler-Maruyama method
- ✅ Derive the unit-variance uniform distribution
- ✅ Justify the √dt normalization factor
- ✅ Interpret the MSD slope (α)
- ✅ Explain convergence criteria
- ✅ Discuss the Central Limit Theorem
- ✅ Walk through any code segment
- ✅ Justify parameter choices

---

## 📚 Additional Resources

### Theoretical Background

**Brownian Motion:**
- Einstein (1905) - "On the Movement of Small Particles"
- Perrin (1909) - Experimental verification

**Stochastic Calculus:**
- Øksendal - "Stochastic Differential Equations"
- Gardiner - "Stochastic Methods"

**Numerical Methods:**
- Kloeden & Platen - "Numerical Solution of SDEs"

### Code Examples

```python
# Quick test: Verify unit variance
import numpy as np

# Gaussian
zeta_gauss = np.random.normal(0, 1, 100000)
print(f"Gaussian variance: {np.var(zeta_gauss):.4f}")  # Should be ≈1.0

# Uniform
zeta_unif = np.random.uniform(-np.sqrt(3), np.sqrt(3), 100000)
print(f"Uniform variance: {np.var(zeta_unif):.4f}")   # Should be ≈1.0
```

### Useful Marimo Commands

```bash
# List all marimo commands
marimo --help

# Tutorial notebooks
marimo tutorial intro
marimo tutorial plots
marimo tutorial dataframes

# Export options
marimo export html notebook.py
marimo export md notebook.py

# Convert from Jupyter
marimo convert notebook.ipynb > notebook.py
```

---

## 🤝 Getting Help

### If You're Stuck

1. **Check this README** - Most common issues covered
2. **Read error messages** - They usually tell you what's wrong
3. **Search the code** - Look for similar examples
4. **Ask your group** - Collaboration is encouraged!
5. **Office hours** - Reach out to Prof. Picardo or TAs

### Useful Links

- **Marimo Docs:** https://docs.marimo.io
- **NumPy Docs:** https://numpy.org/doc/
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/
- **Course Repository:** https://github.com/sithtsar/CL677

---

## 📄 License

MIT License - For educational purposes

---

## 👥 Authors

**Sarthak Mishra** (22b0432) - [GitHub](https://github.com/sithtsar)
**Pratyush Ranjan** (22b0326)

---

## 🔗 Quick Links

- [Repository Root](../)
- [Main README](../README.md)
- [Marimo Documentation](https://docs.marimo.io)
- [Course Materials](https://github.com/sithtsar/CL677)

---

**Last Updated:** January 27, 2026
**Version:** 1.0 - Submission Ready ✅
