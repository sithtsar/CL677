import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import Libraries"""
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Plot Settings
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    return mo, np, plt, solve_ivp


@app.cell
def _(mo):
    """Title"""
    mo.md(
        r"""
        # CL677: Assignment 2 - Method of Lines Solver
        **Modelling Stochastic and Turbulent Transport**

        * **Sarthak Mishra 22b0432 | Pratyush Ranjan 22b0326**

        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Mathematical Formulation & Numerical Method

    ### 1. The Governing Equations
    We are solving the **Fokker-Planck Equation** (FPE) for the probability density function $p(x,t)$.

    **A. Constant Drift (Questions A & C):**
    $\frac{\partial p}{\partial t} = -A \frac{\partial p}{\partial x} + \frac{D}{2} \frac{\partial^2 p}{\partial x^2}$
    $
    * $A$: Drift velocity (advection).
    * $D$: Diffusion coefficient.

    **B. Harmonic Potential (Question B):**
    For a particle in a trap $U(x) = \frac{1}{2}ax^2$, the drift is $v(x) = -ax$.

    $\frac{\partial p}{\partial t} = \frac{\partial}{\partial x}(ax p) + \frac{D}{2} \frac{\partial^2 p}{\partial x^2}
    $

    ---

    ### 2. Numerical Scheme: Method of Lines
    We discretize space into $N$ points with spacing $\Delta x$ but leave time continuous. This converts the PDE into a system of Ordinary Differential Equations (ODEs) solved by `scipy.integrate.solve_ivp` (Runge-Kutta 45).

    **Spatial Discretization (Central Differences):**
    * **First Derivative:** $\frac{\partial p}{\partial x} \approx \frac{p_{i+1} - p_{i-1}}{2\Delta x}$
    * **Second Derivative:** $\frac{\partial^2 p}{\partial x^2} \approx \frac{p_{i+1} - 2p_i + p_{i-1}}{\Delta x^2}$

    ---

    ### 3. Boundary Conditions & Ghost Nodes
    We implement boundary conditions using **Ghost Nodes** ($p_{-1}$ and $p_{N}$) just outside the domain to preserve 2nd-order accuracy.

    **A. Reflecting Boundary (No-Flux) at Walls**
    The probability flux $J$ must be zero at the wall.
    $J(x,t) = v(x)p - \frac{D}{2}\frac{\partial p}{\partial x} = 0 \implies \frac{\partial p}{\partial x} = \frac{2v(x)}{D}p$

    * **For Constant Drift ($v=A$):**
        $\frac{p_{1} - p_{-1}}{2\Delta x} = \frac{2A}{D} p_0 \implies p_{-1} = p_{1} - \frac{4 A \Delta x}{D} p_0$
    * **For Harmonic Trap ($v=-ax$):**
        $\frac{p_{1} - p_{-1}}{2\Delta x} = -\frac{2ax_0}{D} p_0 \implies p_{-1} = p_{1} + \frac{4 a x_0 \Delta x}{D} p_0$

    **B. Absorbing Boundary (Question C)**
    At the right wall ($x=L/2$), particles are removed.
    $p(L/2, t) = 0$
    This is implemented by strictly forcing $p_{N-1} = 0$ at every time step.
    """)
    return


@app.cell
def _(mo):
    """GLOBAL PARAMETERS (Shared across all questions)"""
    mo.md("### 🌍 Global Grid & Time Parameters")

    L_global = mo.ui.slider(5, 50, value=10, label="Domain Length (L)", step=1)
    N_global = mo.ui.slider(50, 500, value=100, label="Grid Points (N)", step=10)
    T_global = mo.ui.slider(1, 100, value=10, label="Maximum Time (T)", step=1)

    global_ui = mo.hstack([L_global, N_global, T_global], justify="center")

    return L_global, N_global, T_global, global_ui


@app.cell
def _(global_ui, mo):
    mo.vstack([global_ui])
    return


@app.cell
def _(np):
    """Helper: Initial Condition"""
    def get_initial_condition(x, l_width):
        p0 = np.zeros_like(x)
        dx = x[1] - x[0]
        mask = (x >= -l_width/2) & (x <= l_width/2)
        if np.sum(mask) > 0:
            p0[mask] = 1.0 / (np.sum(mask) * dx)
        return p0
    return (get_initial_condition,)


@app.cell
def _(np):
    """Helper: FP Equation RHS"""
    def fp_rhs(t, p, x, dx, D, drift_type, A_const, a_harm, boundary_right):
        N = len(p)
        dpdt = np.zeros(N)

        if drift_type == 'constant':
            dpdt[1:-1] = (-A_const * (p[2:] - p[:-2]) / (2*dx) + 
                          (D/2) * (p[2:] - 2*p[1:-1] + p[:-2]) / (dx**2))

            if D > 1e-10:
                p_ghost_left = p[1] - (4*dx*A_const/D)*p[0]
                dpdt[0] = (-A_const * (p[1] - p_ghost_left) / (2*dx) +
                           (D/2) * (p[1] - 2*p[0] + p_ghost_left) / (dx**2))
            else:
                dpdt[0] = -A_const * (p[1] - p[0]) / dx

            if boundary_right == 'absorbing':
                dpdt[N-1] = 0
            else:
                if D > 1e-10:
                    p_ghost_right = p[N-2] + (4*dx*A_const/D)*p[N-1]
                    dpdt[N-1] = (-A_const * (p_ghost_right - p[N-2]) / (2*dx) +
                                 (D/2) * (p_ghost_right - 2*p[N-1] + p[N-2]) / (dx**2))
                else:
                    dpdt[N-1] = -A_const * (p[N-1] - p[N-2]) / dx

        elif drift_type == 'harmonic':
            flux_plus = a_harm * x[2:] * p[2:]
            flux_minus = a_harm * x[:-2] * p[:-2]
            dpdt[1:-1] = ((flux_plus - flux_minus) / (2*dx) +
                          (D/2) * (p[2:] - 2*p[1:-1] + p[:-2]) / (dx**2))

            if D > 1e-10:
                p_ghost_left = p[1] + (4*dx*a_harm*x[0]/D)*p[0]
                flux_ghost = a_harm * (x[0]-dx) * p_ghost_left
                flux_next = a_harm * x[1] * p[1]
                dpdt[0] = ((flux_next - flux_ghost) / (2*dx) + 
                           (D/2) * (p[1] - 2*p[0] + p_ghost_left) / (dx**2))

                p_ghost_right = p[N-2] - (4*dx*a_harm*x[N-1]/D)*p[N-1]
                flux_prev = a_harm * x[N-2] * p[N-2]
                flux_ghost_r = a_harm * (x[N-1]+dx) * p_ghost_right
                dpdt[N-1] = ((flux_ghost_r - flux_prev) / (2*dx) + 
                             (D/2) * (p_ghost_right - 2*p[N-1] + p[N-2]) / (dx**2))

        return dpdt
    return (fp_rhs,)


@app.cell
def _(fp_rhs, get_initial_condition, np, solve_ivp):
    """Helper: Solver with t_eval"""
    def solve_fp_equation(N, L, T_max, D, drift_type, A_const=0, a_harm=0, 
                          l_width=1.0, boundary_right='reflecting', t_points=None):
        x = np.linspace(-L/2, L/2, N)
        dx = L / (N - 1)
        p0 = get_initial_condition(x, l_width)
        if boundary_right == 'absorbing': p0[-1] = 0.0

        if t_points is not None:
            t_eval = t_points
            T_run = max(T_max, t_points[-1])
        else:
            t_eval = np.linspace(0, T_max, 100)
            T_run = T_max

        sol = solve_ivp(
            fun=fp_rhs, t_span=(0, T_run), y0=p0, t_eval=t_eval,
            args=(x, dx, D, drift_type, A_const, a_harm, boundary_right),
            method='RK45', rtol=1e-6, atol=1e-8
        )
        return sol, x, dx
    return (solve_fp_equation,)


@app.cell
def _(np, plt, solve_fp_equation):
    """Plotting Functions"""

    # A1 Plotter
    def plot_A1(N, L, T_max, D, A, l_width):
        times_early = np.arange(0, min(2.0, T_max) + 0.05, 0.2)
        times_long = np.arange(0, T_max + 0.1, 1.0)
        all_times = np.unique(np.concatenate((times_early, times_long)))
        sol, x, _ = solve_fp_equation(N, L, T_max, D, 'constant', A_const=A, l_width=l_width, t_points=all_times)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for idx in np.where(np.isin(sol.t, times_early))[0]:
            if any(np.isclose(sol.t[idx], times_early)):
                ax1.plot(x, sol.y[:, idx], label=f't={sol.t[idx]:.1f}')
        ax1.set_title(f'Early-Time (A={A})'); ax1.legend(ncol=2, fontsize=8)

        cmap = plt.cm.viridis
        idxs = np.where(np.isin(sol.t, times_long))[0]
        for i, idx in enumerate(idxs):
            ax2.plot(x, sol.y[:, idx], color=cmap(i/len(idxs)), label=f't={sol.t[idx]:.1f}')
        ax2.set_title('Long-Time'); ax2.legend(ncol=2, fontsize=8)
        return fig

    # A2 Plotter
    def plot_A2(N, L, T_max, D, A, l_width):
        targets = np.array([0.0, 0.4, 1.0, 2.0, 5.0, T_max])
        targets = targets[targets <= T_max]
        sol, x, _ = solve_fp_equation(N, L, T_max, D, 'constant', A_const=A, l_width=l_width, t_points=targets)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, idx in enumerate(range(len(sol.t))):
            if i >= 6: break
            p = sol.y[:, idx]
            mean = np.trapezoid(x*p, x); var = np.trapezoid((x-mean)**2 * p, x); sigma = np.sqrt(var)
            gauss = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/sigma)**2) if sigma>0 else np.zeros_like(x)
            axes[i].plot(x, p, 'b-', label='Sim'); axes[i].plot(x, gauss, 'r--', label='Gauss')
            axes[i].set_title(f't={sol.t[idx]:.1f}, σ²={var:.2f}')
        plt.tight_layout()
        return fig

    # A4 Plotter
    def plot_A4(N, L, T_max, D, A):
        times = np.linspace(0, T_max, 8)
        sol1, x, _ = solve_fp_equation(N, L, T_max, D, 'constant', A_const=A, l_width=1.0, t_points=times)
        sol2, _, _ = solve_fp_equation(N, L, T_max, D, 'constant', A_const=A, l_width=L, t_points=times)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i in range(len(times)):
            alpha = 0.3 + 0.7*(i/len(times))
            axes[0].plot(x, sol1.y[:, i], 'b-', alpha=alpha)
            axes[1].plot(x, sol2.y[:, i], 'r-', alpha=alpha)
        axes[0].set_title(f'Pulse IC, A={A}'); axes[1].set_title(f'Uniform IC, A={A}')

        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, sol1.y[:, -1], 'b-', lw=3, alpha=0.6, label='Pulse Final')
        ax.plot(x, sol2.y[:, -1], 'r--', lw=2, label='Uniform Final')
        ax.set_title('Final State Comparison'); ax.legend()
        return fig, fig2

    # B Plotter
    def plot_B(N, L, T_max, D, l_width):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        times = np.linspace(0, T_max, 10); max_p = 0
        for i, a in enumerate([0.2, 0.5, 1.0]):
            sol, x, _ = solve_fp_equation(N, L, T_max, D, 'harmonic', a_harm=a, l_width=l_width, t_points=times)
            for j, idx in enumerate(range(len(sol.t))):
                color = plt.cm.plasma(j/len(sol.t))
                p = sol.y[:, idx]; axes[i].plot(x, p, color=color, alpha=0.7); max_p = max(max_p, np.max(p))
            axes[i].set_title(f'Trap a={a}')
        for ax in axes: ax.set_ylim(-0.02, max_p*1.1)
        return fig

    # C Plotter
    def plot_C(N, L, T_max, D):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)); colors = ['blue', 'green', 'red']
        for A_val, color in zip([0.0, 0.2, 1.0], colors):
            sol, x, _ = solve_fp_equation(N, L, T_max, D, 'constant', A_const=A_val, boundary_right='absorbing', l_width=1.0)
            P_t = np.trapezoid(sol.y, x, axis=0)
            axes[0].plot(sol.t, P_t, color=color, label=f'A={A_val}')
            mask = P_t > 1e-10
            axes[1].semilogy(sol.t[mask], P_t[mask], color=color)
            if np.sum(mask) > 0: axes[2].loglog(sol.t[mask], P_t[mask], color=color)
        axes[0].legend()
        return fig
    return plot_A1, plot_A2, plot_A4, plot_B, plot_C


@app.cell
def _(mo):
    mo.md("""
    ## 1. Question A: Constant Drift & Diffusion
    """)
    return


@app.cell
def _(mo):
    # === SLIDERS SPECIFIC TO Q A1 ===
    mo.md("#### **Settings for A1 (Evolution)**")
    D_A1 = mo.ui.number(0.1, 5.0, value=1.0, step=0.1, label="Diffusion (D)")
    l_A1 = mo.ui.number(0.1, 10.0, value=1.0, step=0.1, label="Pulse Width (ℓ)")
    A_A1 = mo.ui.number(-2.0, 2.0, value=0.0, step=0.1, label="Drift (A)")

    ui_A1 = mo.hstack([D_A1, l_A1, A_A1], justify="start")
    return A_A1, D_A1, l_A1, ui_A1


@app.cell
def _(A_A1, D_A1, L_global, N_global, T_global, l_A1, mo, plot_A1, ui_A1):
    fig_A1 = plot_A1(N_global.value, L_global.value, T_global.value, D_A1.value, A_A1.value, l_A1.value)
    mo.vstack([ui_A1, fig_A1])
    return


@app.cell
def _(mo):
    # === SLIDERS SPECIFIC TO Q A2 ===
    mo.md("#### **Settings for A2 (Gaussian Fit)**")
    D_A2 = mo.ui.number(0.1, 5.0, value=1.0, step=0.1, label="Diffusion (D)")
    l_A2 = mo.ui.number(0.1, 10.0, value=1.0, step=0.1, label="Pulse Width (ℓ)")
    A_A2 = mo.ui.number(-2.0, 2.0, value=0.0, step=0.1, label="Drift (A)")

    ui_A2 = mo.hstack([D_A2, l_A2, A_A2], justify="start")
    return A_A2, D_A2, l_A2, ui_A2


@app.cell
def _(A_A2, D_A2, L_global, N_global, T_global, l_A2, mo, plot_A2, ui_A2):
    fig_A2 = plot_A2(N_global.value, L_global.value, T_global.value, D_A2.value, A_A2.value, l_A2.value)
    mo.vstack([ui_A2, fig_A2])
    return


@app.cell
def _(mo):
    # === SLIDERS SPECIFIC TO Q A4 ===
    mo.md("#### **Settings for A4 (Drift Comparison)**")
    D_A4 = mo.ui.number(0.1, 5.0, value=1.0, step=0.1, label="Diffusion (D)")
    A_A4 = mo.ui.number(0.1, 2.0, value=1.0, step=0.1, label="Drift (A)")

    ui_A4 = mo.hstack([D_A4, A_A4], justify="start")
    return A_A4, D_A4, ui_A4


@app.cell
def _(A_A4, D_A4, L_global, N_global, T_global, mo, plot_A4, ui_A4):
    fig_A4_1, fig_A4_2 = plot_A4(N_global.value, L_global.value, T_global.value, D_A4.value, A_A4.value)
    mo.vstack([ui_A4, fig_A4_1, fig_A4_2])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Analysis: Constant Drift & Diffusion

    **A1. Early vs. Long-Time Behavior**
    * **Early Time ($t < 2.0$):** The sharp corners of the initial square pulse smoothen out rapidly. This is because the diffusion term $\frac{D}{2}\frac{\partial^2 p}{\partial x^2}$ is largest where the curvature is highest (the corners).
    * **Long Time ($t \to \infty$):** With **Reflecting Boundaries** and zero drift ($A=0$), the particles eventually fill the box evenly. The profile flattens into a **Uniform Distribution** $p(x) \approx 1/L$.

    **A2. Gaussian Convergence**
    * The pulse evolves into a Gaussian shape very quickly ($t \approx 0.5$).
    * **Variance Growth:** For a free particle, variance grows as $\sigma^2 = 2Dt$.
    * **Boundary Effect:** At late times, the simulation **departs** from the theoretical Gaussian fit because the box is finite. Particles that would have spread to infinity are reflected back, causing the "tails" of the distribution to fold in and flatten the curve.

    **A3. Steady State ($t \to \infty$)**
    * **Physical Meaning:** The final uniform distribution ($A=0$) represents maximum entropy. Without an external force to bias the motion, the particle is equally likely to be found anywhere in the domain.

    **A4. Effect of Drift ($A=1$)**
    * With $A=1$, the steady state is **not uniform**. Drift pushes particles to the right, creating an exponential profile $p(x) \propto e^{v x/D}$ against the boundary.
    * **Ergodicity:** As shown in the comparison plot, the **Pulse IC** and **Uniform IC** reach the **exact same final state**. This proves that the final equilibrium depends only on the parameters ($A, D$), not on the initial condition.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Question B: Harmonic Potential
    """)
    return


@app.cell
def _(mo):
    # === SLIDERS SPECIFIC TO Q B ===
    mo.md("#### **Settings for Question B**")
    D_B = mo.ui.number(0.1, 5.0, value=1.0, step=0.1, label="Diffusion (D)")
    l_B = mo.ui.number(0.1, 5.0, value=2.0, step=0.1, label="Initial Width (ℓ)")

    ui_B = mo.hstack([D_B, l_B], justify="start")
    return D_B, l_B, ui_B


@app.cell
def _(D_B, L_global, N_global, T_global, l_B, mo, plot_B, ui_B):
    fig_B = plot_B(N_global.value, L_global.value, T_global.value, D_B.value, l_B.value)
    mo.vstack([ui_B, fig_B])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Analysis: Harmonic Potential (Optical Trap)

    **Physics of the Trap**
    The harmonic potential $U(x) = \frac{1}{2}ax^2$ creates a restoring drift velocity $v = -ax$ that pulls particles toward the center ($x=0$). This competes with Diffusion ($D$), which tries to spread them out.

    **Steady State Form:**
    The final distribution is the **Boltzmann Distribution**:
    $p_{ss}(x) = C \cdot \exp\left(-\frac{U(x)}{k_B T_{eff}}\right) \propto \exp\left(-\frac{a x^2}{D}\right)$
    This is a Gaussian centered at 0. The width (variance) is controlled by the ratio **D/a**.

    **Observations:**
    * **Weak Trap ($a=0.2$):** Diffusion dominates, resulting in a broad distribution.
    * **Strong Trap ($a=1.0$):** The restoring force dominates, pinning particles in a narrow peak near the center.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Question C: Absorbing Boundary
    """)
    return


@app.cell
def _(mo):
    # === SLIDERS SPECIFIC TO Q C ===
    mo.md("#### **Settings for Question C**")
    D_C = mo.ui.number(0.1, 5.0, value=1.0, step=0.1, label="Diffusion (D)")

    ui_C = mo.hstack([D_C], justify="start")
    return D_C, ui_C


@app.cell
def _(D_C, L_global, N_global, T_global, mo, plot_C, ui_C):
    fig_C = plot_C(N_global.value, L_global.value, T_global.value, D_C.value)
    mo.vstack([ui_C, fig_C])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📝 Analysis: Absorbing Boundary & Escape Probability

    In this scenario, the right boundary is an **Absorber** ($p=0$). We track the total remaining probability $P(t) = \int p(x,t) dx$ to see how fast particles escape.

    **Semi-Log Plot:**
    The curves (especially for $A=1$) become straight lines at long times. A straight line on a semi-log plot indicates **Exponential Decay**:
    $P(t) \sim e^{-k t}$
    This confirms that the escape process becomes a "first-order" rate process once the initial transients die out.

    **Effect of Drift:**
    * **$A=0$:** Pure diffusion. Escape is slow and gradual.
    * **$A=1$:** Strong drift toward the absorber. The probability drops rapidly as particles are actively pushed into the trap.
    """)
    return


if __name__ == "__main__":
    app.run()
