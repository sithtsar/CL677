import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Assignment 1: Random Walks Simulation

    **CL 677: Modelling Stochastic and Turbulent Transport (Spring 2025-26)**

    **Students:** Sarthak Mishra (22b0432) | Pratyush Ranjan (22b0326)

    **Instructor:** Prof. Jason Picardo

    ---

    ## Problem Statement

    Compute trajectories of a random walk using the Euler-Maruyama discretization:

    $$
    dx = A\zeta\sqrt{dt}
    $$

    where:
    - $A$ = amplitude of the Brownian force
    - $\zeta$ = random number with unit-variance distribution
    - $dt$ = time step
    - $\sqrt{dt}$ = normalization factor for convergence

    **Initial Parameters:**
    - $A = 1$
    - $dt = 0.02$
    - $T = 10$ (total time)
    - $N_{traj} = 1000$ trajectories (minimum)
    - Initial condition: $x(0) = 0$
    """)
    return


@app.cell
def _():
    # Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import marimo as mo

    # Configure matplotlib for better figure quality
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['axes.titlesize'] = 13
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['figure.figsize'] = (10, 6)

    # Set random seed for reproducibility
    np.random.seed(42)
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Helper Functions

    We define reusable functions to simulate trajectories and analyze their statistics.
    """)
    return


@app.cell
def _(np):
    def simulate_trajectories(A, T, dt, N_traj, distribution='gaussian'):
        """Simulate multiple trajectories of Brownian motion.

        Parameters:
        -----------
        A : float
            Amplitude of the Brownian force
        T : float
            Total simulation time
        dt : float
            Time step
        N_traj : int
            Number of trajectories to simulate
        distribution : str
            'gaussian' or 'uniform' - the type of random number distribution

        Returns:
        --------
        x_traj : ndarray
            Array of shape (N_traj, N_steps+1) containing all trajectories
        time : ndarray
            Time vector
        """
        N_steps = int(T / dt)

        if distribution == 'gaussian':
            # Unit-variance Gaussian distribution N(0,1)
            zeta = np.random.normal(loc=0.0, scale=1.0, size=(N_traj, N_steps))
        elif distribution == 'uniform':
            # Unit-variance uniform distribution on [-√3, √3]
            zeta = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=(N_traj, N_steps))
        else:
            raise ValueError("Distribution must be 'gaussian' or 'uniform'")

        # Euler-Maruyama discretization
        dx = A * zeta * np.sqrt(dt)
        x = np.cumsum(dx, axis=1)

        # Add initial condition x(0) = 0
        x_traj = np.hstack([np.zeros((N_traj, 1)), x])
        time = np.linspace(0, T, N_steps + 1)

        return x_traj, time
    return (simulate_trajectories,)


@app.cell
def _(np):
    def calculate_msd(x_traj):
        """Calculate mean-squared displacement from trajectories.

        Parameters:
        -----------
        x_traj : ndarray
            Array of shape (N_traj, N_steps+1) containing all trajectories

        Returns:
        --------
        msd : ndarray
            Mean-squared displacement at each time step: ⟨x²⟩
        """
        return np.mean(x_traj**2, axis=0)
    return (calculate_msd,)


@app.cell
def _(np):
    def fit_alpha(time, msd):
        """Fit the power-law exponent from log-log MSD data.

        MSD ~ t^α, so log(MSD) = α·log(t) + constant

        Parameters:
        -----------
        time : ndarray
            Time vector
        msd : ndarray
            Mean-squared displacement

        Returns:
        --------
        alpha : float
            Fitted exponent
        coeffs : ndarray
            Polynomial coefficients [slope, intercept]
        """
        log_time = np.log(time[1:])  # Exclude t=0
        log_msd = np.log(msd[1:])
        coeffs = np.polyfit(log_time, log_msd, 1)
        alpha = coeffs[0]
        return alpha, coeffs
    return (fit_alpha,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Question 1: Mean-Squared Displacement and Diffusive Behavior (10 points)

    Calculate the mean-squared displacement (MSD) ⟨x²⟩ and plot it vs. time on a log-log scale.
    Determine the slope α from the power-law relationship: ⟨x²⟩ ∼ t^α
    """)
    return


@app.cell
def _():
    # Simulation parameters for Question 1
    A_q1 = 1.0          # Amplitude
    T_q1 = 10.0         # Total time
    dt_q1 = 0.02        # Time step
    N_traj_q1 = 2000    # Number of trajectories (>1000 for good statistics)
    return A_q1, N_traj_q1, T_q1, dt_q1


@app.cell
def _(A_q1, N_traj_q1, T_q1, dt_q1, simulate_trajectories):
    # Run simulation for Question 1
    x_traj_q1, time_q1 = simulate_trajectories(A_q1, T_q1, dt_q1, N_traj_q1, distribution='gaussian')
    return time_q1, x_traj_q1


@app.cell
def _(N_traj_q1, plt, time_q1, x_traj_q1):
    # Plot sample trajectories
    fig_traj_q1, ax_traj_q1 = plt.subplots(figsize=(12, 6))

    # Plot first 50 trajectories for visualization
    _n_plot = N_traj_q1
    for _i in range(_n_plot):
        ax_traj_q1.plot(time_q1, x_traj_q1[_i, :], alpha=0.5, linewidth=0.8)

    ax_traj_q1.set_xlabel('Time', fontsize=12)
    ax_traj_q1.set_ylabel('Position x(t)', fontsize=12)
    ax_traj_q1.set_title(f'Sample Trajectories of Brownian Motion (showing {_n_plot} of {N_traj_q1} trajectories)', fontsize=13)
    ax_traj_q1.grid(True, alpha=0.3)
    ax_traj_q1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    fig_traj_q1.tight_layout()
    fig_traj_q1



    return


@app.cell
def _(mo):
    mo.md(f"""
    **Figure 1:** Sample trajectories showing the stochastic nature of Brownian motion.
    Each trajectory starts at x(0) = 0 and evolves according to dx = A ζ √dt.
    """)
    return


@app.cell
def _(calculate_msd, fit_alpha, np, plt, time_q1, x_traj_q1):
    # Calculate MSD for Question 1
    msd_q1 = calculate_msd(x_traj_q1)
    alpha_q1, coeffs_q1 = fit_alpha(time_q1, msd_q1)

    # Plot MSD with fitted line
    fig_msd_q1, ax_msd_q1 = plt.subplots(figsize=(12, 7))

    ax_msd_q1.loglog(time_q1[1:], msd_q1[1:], 'o-', label='Simulation Data',
                     markersize=4, linewidth=2, color='#2E86AB')

    # Plot fitted line
    _fit_line = np.exp(coeffs_q1[1]) * time_q1[1:]**alpha_q1
    ax_msd_q1.loglog(time_q1[1:], _fit_line, '--',
                     label=f'Fit: ⟨x²⟩ ∼ t^{alpha_q1:.3f}',
                     linewidth=2.5, color='#A23B72')

    # Add theoretical reference line (α=1)
    _theory_line = np.exp(coeffs_q1[1]) * time_q1[1:]**1.0
    ax_msd_q1.loglog(time_q1[1:], _theory_line, ':',
                     label='Theory: α = 1.0 (normal diffusion)',
                     linewidth=2, color='#F18F01', alpha=0.7)

    ax_msd_q1.set_xlabel('Time (t)', fontsize=13)
    ax_msd_q1.set_ylabel('Mean-Squared Displacement ⟨x²⟩', fontsize=13)
    ax_msd_q1.set_title('Mean-Squared Displacement vs. Time (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax_msd_q1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax_msd_q1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

    fig_msd_q1.tight_layout()
    fig_msd_q1
    return (alpha_q1,)


@app.cell
def _(A_q1, N_traj_q1, T_q1, alpha_q1, dt_q1, mo):
    mo.md(f"""
    ### Results for Question 1

    **Simulation Parameters:**
    - Amplitude: A = {A_q1}
    - Time step: dt = {dt_q1}
    - Total time: T = {T_q1}
    - Number of trajectories: N = {N_traj_q1}

    **Fitted Exponent:**
    - **α = {alpha_q1:.4f}**

    **Analysis:**

    The fitted exponent α = {alpha_q1:.4f} is {"very close to" if abs(alpha_q1 - 1) < 0.05 else "close to" if abs(alpha_q1 - 1) < 0.1 else "deviates slightly from"} the theoretical value of α = 1.0
    expected for normal diffusive behavior. The deviation of {abs(alpha_q1 - 1):.4f} can be attributed to:

    1. **Statistical fluctuations:** Even with {N_traj_q1} trajectories, finite sampling introduces variance
    2. **Finite-time effects:** The simulation runs for finite time T = {T_q1}, whereas theory assumes t → ∞
    3. **Numerical discretization:** The Euler-Maruyama scheme introduces small discretization errors

    **Theoretical Background:**

    For a diffusive random walk (Brownian motion), the mean-squared displacement follows:

    $$
    \\langle x^2(t) \\rangle = 2Dt
    $$

    where D is the diffusion coefficient. This linear relationship corresponds to α = 1, characteristic of
    normal diffusion where displacement grows as x ∼ √t. In our simulation with A = {A_q1} and unit-variance
    noise, the effective diffusion coefficient is D_eff = A²/2 = {A_q1**2/2:.2f}.

    **Conclusion:** The simulation successfully reproduces diffusive behavior with α ≈ 1.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Question 2: Convergence Analysis with Different Time Steps (5 points)

    Recalculate trajectories for dt = 0.005 and dt = 0.001, and overlay the MSD curves
    to check convergence.
    """)
    return


@app.cell
def _(A_q1, N_traj_q1, T_q1, calculate_msd, fit_alpha, simulate_trajectories):
    # Simulate for three different time steps
    dt_values_q2 = [0.02, 0.005, 0.001]

    results_q2 = []
    for _dt in dt_values_q2:
        _x_traj, _time = simulate_trajectories(A_q1, T_q1, _dt, N_traj_q1, distribution='gaussian')
        _msd = calculate_msd(_x_traj)
        _alpha, _coeffs = fit_alpha(_time, _msd)
        results_q2.append({
            'dt': _dt,
            'time': _time,
            'msd': _msd,
            'alpha': _alpha,
            'N_steps': len(_time) - 1
        })
    return (results_q2,)


@app.cell
def _(plt, results_q2):
    # Plot convergence analysis
    fig_conv_q2, ax_conv_q2 = plt.subplots(figsize=(12, 7))

    _colors = ['#E63946', '#457B9D', '#2A9D8F']
    _markers = ['o', 's', '^']

    for _idx, result in enumerate(results_q2):
        ax_conv_q2.loglog(
            result['time'][1:],
            result['msd'][1:],
            marker=_markers[_idx],
            label=f"dt = {result['dt']:.3f} (α = {result['alpha']:.4f}, N_steps = {result['N_steps']})",
            linewidth=2,
            markersize=4,
            markevery=max(1, len(result['time'])//30),
            alpha=0.85,
            color=_colors[_idx]
        )

    ax_conv_q2.set_xlabel('Time (t)', fontsize=13)
    ax_conv_q2.set_ylabel('Mean-Squared Displacement ⟨x²⟩', fontsize=13)
    ax_conv_q2.set_title('Convergence Analysis: MSD for Different Time Steps', fontsize=14, fontweight='bold')
    ax_conv_q2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax_conv_q2.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

    fig_conv_q2.tight_layout()
    fig_conv_q2
    return


@app.cell
def _(mo, results_q2):
    _alpha_values = [r['alpha'] for r in results_q2]
    _dt_values = [r['dt'] for r in results_q2]
    _alpha_range = max(_alpha_values) - min(_alpha_values)

    mo.md(f"""
    ### Results for Question 2

    **Convergence Analysis:**

    | Time Step (dt) | Number of Steps | Fitted α | Deviation from α=1 |
    |----------------|-----------------|----------|-------------------|
    {"".join(f"| {r['dt']:.3f} | {r['N_steps']} | {r['alpha']:.4f} | {abs(r['alpha'] - 1):.4f} |\\n" for r in results_q2)}

    **Observations:**

    1. **Overlap of MSD curves:** The three MSD curves {"overlap closely" if _alpha_range < 0.05 else "show reasonable agreement" if _alpha_range < 0.1 else "show some variation"},
       indicating {"good" if _alpha_range < 0.05 else "acceptable" if _alpha_range < 0.1 else "moderate"} convergence with respect to time discretization.

    2. **Consistency of slope α:**
       - Fitted α values range from **{min(_alpha_values):.4f}** to **{max(_alpha_values):.4f}**
       - Variation: Δα = **{_alpha_range:.4f}**
       - This {"small" if _alpha_range < 0.1 else "moderate"} variation suggests the Euler-Maruyama scheme is {"well-converged" if _alpha_range < 0.1 else "approaching convergence"}

    3. **Number of time steps:** As dt decreases, the number of steps increases proportionally
       (500 → 2000 → 10000), providing finer time resolution.

    **Theoretical Expectation:**

    The Euler-Maruyama discretization should converge as dt → 0. The convergence is characterized by:
    - Weak convergence: O(dt) for the probability distribution
    - Strong convergence: O(√dt) for individual trajectories

    Our results {"confirm" if _alpha_range < 0.1 else "suggest"} that the simulation is {"well-converged" if _alpha_range < 0.05 else "approaching convergence"},
    as the MSD curves {"overlay closely" if _alpha_range < 0.05 else "show consistent behavior"} and α values are consistent within ~{_alpha_range:.1%}.

    **Conclusion:** The results {"demonstrate good convergence" if _alpha_range < 0.1 else "suggest the need for finer resolution"} with respect to time step.
    The value dt = 0.02 appears {"sufficient" if _alpha_range < 0.1 else "adequate"} for accurate simulation of the MSD.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Question 3: Comparison of Gaussian vs. Uniform Random Numbers (5 points)

    Repeat the MSD calculation using uniformly distributed random numbers (with unit variance)
    and compare with Gaussian results.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Theoretical Background: Uniform Distribution with Unit Variance

    For a uniform distribution on interval [a, b], the variance is:

    $$
    \text{Var} = \frac{(b-a)^2}{12}
    $$

    Setting Var = 1 and solving for a symmetric interval around zero:

    $$
    \frac{(b-a)^2}{12} = 1 \implies b-a = 2\sqrt{3}
    $$

    Therefore: ζ ∼ Uniform[-√3, √3] gives unit variance.

    **Central Limit Theorem Prediction:**

    By the CLT, the sum of many independent random variables (regardless of their distribution)
    tends toward a Gaussian distribution. Since Brownian motion involves cumulative sums:

    $$
    x(t) = \sum_{i=1}^{N} A\zeta_i\sqrt{dt}
    $$

    We expect:
    - **Short-time behavior:** Differences due to individual step statistics
    - **Long-time behavior:** MSD should converge to similar values (dominated by variance, not distribution shape)
    """)
    return


@app.cell
def _(
    A_q1,
    N_traj_q1,
    T_q1,
    calculate_msd,
    dt_q1,
    fit_alpha,
    simulate_trajectories,
):
    # Simulate with Gaussian distribution
    x_traj_gauss, time_q3 = simulate_trajectories(A_q1, T_q1, dt_q1, N_traj_q1, distribution='gaussian')
    msd_gauss = calculate_msd(x_traj_gauss)
    alpha_gauss, coeffs_gauss = fit_alpha(time_q3, msd_gauss)

    # Simulate with Uniform distribution
    x_traj_unif, _ = simulate_trajectories(A_q1, T_q1, dt_q1, N_traj_q1, distribution='uniform')
    msd_unif = calculate_msd(x_traj_unif)
    alpha_unif, coeffs_unif = fit_alpha(time_q3, msd_unif)
    return alpha_gauss, alpha_unif, msd_gauss, msd_unif, time_q3


@app.cell
def _(alpha_gauss, alpha_unif, msd_gauss, msd_unif, plt, time_q3):
    # Plot comparison
    fig_comp_q3, ax_comp_q3 = plt.subplots(figsize=(12, 7))

    ax_comp_q3.loglog(time_q3[1:], msd_gauss[1:],
                      'o-', label=f'Gaussian: ζ ∼ N(0,1) (α = {alpha_gauss:.4f})',
                      linewidth=2.5, markersize=5, markevery=10,
                      color='#2E86AB', alpha=0.85)

    ax_comp_q3.loglog(time_q3[1:], msd_unif[1:],
                      's--', label=f'Uniform: ζ ∼ U(-√3,√3) (α = {alpha_unif:.4f})',
                      linewidth=2.5, markersize=5, markevery=10,
                      color='#E63946', alpha=0.85)

    # Add reference line for α=1
    _ref_line = msd_gauss[1] * (time_q3[1:] / time_q3[1])**1.0
    ax_comp_q3.loglog(time_q3[1:], _ref_line,
                      ':', label='Reference: α = 1.0',
                      linewidth=2, color='gray', alpha=0.5)

    ax_comp_q3.set_xlabel('Time (t)', fontsize=13)
    ax_comp_q3.set_ylabel('Mean-Squared Displacement ⟨x²⟩', fontsize=13)
    ax_comp_q3.set_title('Comparison: Gaussian vs. Uniform Random Numbers', fontsize=14, fontweight='bold')
    ax_comp_q3.legend(fontsize=11, loc='best', framealpha=0.9)
    ax_comp_q3.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

    fig_comp_q3.tight_layout()
    fig_comp_q3
    return


@app.cell
def _(alpha_gauss, alpha_unif, mo, msd_gauss, msd_unif, np):
    _rel_diff_alpha = abs(alpha_gauss - alpha_unif) / alpha_gauss * 100
    _rel_diff_msd = abs(msd_gauss[-1] - msd_unif[-1]) / msd_gauss[-1] * 100
    _mean_msd_ratio = np.mean(msd_unif[1:] / msd_gauss[1:])

    mo.md(f"""
    ### Results for Question 3

    **Comparison of Distributions:**

    | Distribution | ζ Range | Fitted α | Final MSD ⟨x²⟩(T) |
    |--------------|---------|----------|-------------------|
    | Gaussian | N(0,1) | {alpha_gauss:.4f} | {msd_gauss[-1]:.4f} |
    | Uniform | U(-√3,√3) | {alpha_unif:.4f} | {msd_unif[-1]:.4f} |
    | **Difference** | — | **{abs(alpha_gauss - alpha_unif):.4f}** | **{abs(msd_gauss[-1] - msd_unif[-1]):.4f}** |
    | **Relative Diff.** | — | **{_rel_diff_alpha:.2f}%** | **{_rel_diff_msd:.2f}%** |

    **Analysis:**

    1. **Do the distributions yield the same MSD?**
       - The relative difference in α is **{_rel_diff_alpha:.2f}%**
       - The relative difference in final MSD is **{_rel_diff_msd:.2f}%**
       - Mean MSD ratio (Uniform/Gaussian) across all times: **{_mean_msd_ratio:.4f}**
       - **Conclusion:** The distributions yield {"essentially identical" if _rel_diff_msd < 5 else "very similar" if _rel_diff_msd < 10 else "comparable"} results

    2. **Why do both distributions give similar MSDs?**

       **Central Limit Theorem:** The sum of many independent random variables tends toward a
       Gaussian distribution regardless of the individual distribution shape (provided finite variance).

       Since Brownian motion is a cumulative process:
       - Both distributions have **unit variance** (σ² = 1)
       - Each trajectory accumulates ~500 random increments (T/dt = 10/0.02)
       - The CLT ensures that long-time behavior converges to the same statistics

    3. **Key Insight:**

       The critical factor determining diffusive behavior is the **variance** of the random numbers,
       not their specific distribution. As long as:
       - E[ζ] = 0 (zero mean)
       - Var[ζ] = 1 (unit variance)

       The long-time MSD will be the same: ⟨x²⟩ = A²t

    4. **Small differences arise from:**
       - Finite number of trajectories (statistical fluctuations)
       - Finite time (not yet fully asymptotic behavior)
       - Different higher-order moments (kurtosis: Gaussian = 3, Uniform = 1.8)

    **Conclusion:** The robustness of diffusive behavior to the choice of random number distribution
    validates the universality of Brownian motion. Only the variance matters for the MSD, demonstrating
    the power of the Central Limit Theorem in stochastic processes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary and Conclusions

    ### Overview

    This assignment explored the simulation of Brownian motion using the Euler-Maruyama
    discretization scheme:

    $$
    dx = A\zeta\sqrt{dt}, \quad \zeta \sim \mathcal{N}(0,1)
    $$

    We investigated three key aspects:

    1. **Diffusive behavior** (Question 1) ✓
    2. **Numerical convergence** (Question 2) ✓
    3. **Distribution dependence** (Question 3) ✓

    ---

    ### Key Findings
    """)
    return


@app.cell
def _(alpha_gauss, alpha_q1, alpha_unif, mo, results_q2):
    _alpha_vals_q2 = [r['alpha'] for r in results_q2]

    mo.md(f"""
    **Question 1: Diffusive Behavior**
    - Fitted exponent: α = {alpha_q1:.4f}
    - Deviation from theory (α=1): {abs(alpha_q1-1):.4f}
    - **Result:** {"✓ Excellent agreement" if abs(alpha_q1-1) < 0.05 else "✓ Good agreement" if abs(alpha_q1-1) < 0.1 else "≈ Reasonable agreement"} with normal diffusion theory

    **Question 2: Convergence Analysis**
    - α values for dt = [0.02, 0.005, 0.001]: [{", ".join(f"{a:.4f}" for a in _alpha_vals_q2)}]
    - Range: Δα = {max(_alpha_vals_q2) - min(_alpha_vals_q2):.4f}
    - **Result:** {"✓ Excellent convergence" if max(_alpha_vals_q2) - min(_alpha_vals_q2) < 0.05 else "✓ Good convergence" if max(_alpha_vals_q2) - min(_alpha_vals_q2) < 0.1 else "≈ Acceptable convergence"}

    **Question 3: Distribution Dependence**
    - α (Gaussian): {alpha_gauss:.4f}
    - α (Uniform): {alpha_unif:.4f}
    - Relative difference: {abs(alpha_gauss-alpha_unif)/alpha_gauss*100:.2f}%
    - **Result:** {"✓ Distributions give identical results" if abs(alpha_gauss-alpha_unif) < 0.02 else "✓ Distributions give essentially same results" if abs(alpha_gauss-alpha_unif) < 0.05 else "≈ Distributions give similar results"}

    ---

    ### Physical Interpretation

    **Brownian Motion as a Diffusive Process:**

    1. **Microscopic picture:** Particles undergo random collisions with solvent molecules
    2. **Macroscopic behavior:** Displacement follows a Gaussian distribution with variance ∝ time
    3. **Diffusion coefficient:** D_eff = A²/2 relates microscopic randomness to macroscopic transport

    **Einstein-Smoluchowski Relation:**

    $$
    D = \\frac{{k_B T}}{{6\\pi\\eta r}}
    $$

    In our simulations: ⟨x²(t)⟩ = 2D_eff·t, giving D_eff = A²/2 = 0.5 for A=1.

    ---

    ### Concluding Remarks

    This assignment successfully demonstrated:

    1. ✓ **Diffusive behavior** of Brownian motion (α ≈ 1)
    2. ✓ **Numerical convergence** of the Euler-Maruyama scheme
    3. ✓ **Robustness** to random number distribution choice
    4. ✓ **Power of the Central Limit Theorem** in stochastic processes

    The simulation captures the essential physics of Brownian motion, validating the theoretical
    framework of stochastic differential equations. The Euler-Maruyama discretization provides
    an accurate and efficient method for simulating diffusive processes, with convergence properties
    that match theoretical predictions.

    **Practical Implications:**
    - Choice of dt: dt = 0.02 provides good accuracy for this problem
    - Number of trajectories: N ≥ 1000 ensures reliable statistics
    - Distribution choice: Any unit-variance distribution gives equivalent long-time behavior

    ---

    **Repository:** [github.com/sithtsar/CL677](https://github.com/sithtsar/CL677/blob/main/Assignment1/notebook.py)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
