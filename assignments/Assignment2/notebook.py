import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import all required libraries"""
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Global Plot Settings for professional appearance
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12

    return mo, np, plt, solve_ivp


@app.cell
def _(mo):
    """Title and Introduction"""
    mo.md(
        r"""
        # CL677: Assignment 2 - Method of Lines Solver
        **Modelling Stochastic and Turbulent Transport**

        - Sarthak Mishra 22b0432 | Pratyush Ranjan 22b0326 
        - Prof. Jason Picardo 

        This notebook implements a Finite Difference solver for the Fokker-Planck equation using the **Method of Lines**.

        ## The Fokker-Planck Equations

        ### Question A & C: Constant Drift
        $$\frac{\partial p}{\partial t} = -A\frac{\partial p}{\partial x} + \frac{D}{2} \frac{\partial^2 p}{\partial x^2}$$

        ### Question B: Harmonic Potential
        $$\frac{\partial p}{\partial t} = \frac{\partial}{\partial x}(pax) + \frac{D}{2} \frac{\partial^2 p}{\partial x^2}$$

        ---
        """
    )
    return


@app.cell
def _(np):
    """Initial condition generator"""
    def get_initial_condition(x, l_width):
        """
        Creates a normalized square pulse of width l_width centered at x=0.

        Args:
            x: spatial grid
            l_width: width of the pulse

        Returns:
            p0: initial probability distribution (normalized to integrate to 1)
        """
        p0 = np.zeros_like(x)
        dx = x[1] - x[0]

        # Square pulse: constant value between -l_width/2 and +l_width/2
        mask = (x >= -l_width/2) & (x <= l_width/2)

        if np.sum(mask) > 0:
            # Normalize so that integral equals 1
            p0[mask] = 1.0 / (np.sum(mask) * dx)

        return p0

    return (get_initial_condition,)


@app.cell
def _(np):
    """RHS function for the Fokker-Planck equation"""
    def fp_rhs(t, p, x, dx, D, drift_type, A_const, a_harm, boundary_right):
        """
        Compute the right-hand side of the FP equation using finite differences.

        This implements the Method of Lines: spatial derivatives are discretized
        using central differences, yielding a system of ODEs in time.

        Boundary conditions are enforced using ghost nodes.

        Args:
            t: current time (required by solve_ivp but not used)
            p: probability distribution at current time
            x: spatial grid
            dx: grid spacing
            D: diffusion coefficient
            drift_type: 'constant' or 'harmonic'
            A_const: constant drift velocity (for constant drift)
            a_harm: harmonic trap strength (for harmonic potential)
            boundary_right: 'reflecting' or 'absorbing'

        Returns:
            dpdt: time derivative of p at all grid points
        """
        N = len(p)
        dpdt = np.zeros(N)

        if drift_type == 'constant':
            # Case A/C: dp/dt = -A * dp/dx + (D/2) * d²p/dx²

            # --- INTERIOR POINTS (i = 1 to N-2) ---
            # Use central differences for both first and second derivatives
            dpdt[1:-1] = (-A_const * (p[2:] - p[:-2]) / (2*dx) + 
                          (D/2) * (p[2:] - 2*p[1:-1] + p[:-2]) / (dx**2))

            # --- LEFT BOUNDARY (i=0): NO-FLUX CONDITION ---
            # Flux J = -A*p + (D/2)*dp/dx = 0
            # Rearranging: dp/dx = (2*A/D)*p
            # Using ghost node p[-1], central difference: (p[1] - p[-1])/(2*dx) = (2*A/D)*p[0]
            # Solve for p[-1]: p[-1] = p[1] - (4*dx*A/D)*p[0]

            if D > 1e-10:  # Avoid division by zero
                p_ghost_left = p[1] - (4*dx*A_const/D)*p[0]
                dpdt[0] = (-A_const * (p[1] - p_ghost_left) / (2*dx) +
                           (D/2) * (p[1] - 2*p[0] + p_ghost_left) / (dx**2))
            else:
                # Pure advection case
                dpdt[0] = -A_const * (p[1] - p[0]) / dx

            # --- RIGHT BOUNDARY (i=N-1) ---
            if boundary_right == 'absorbing':
                # Absorbing BC: p(L/2, t) = 0 for all t
                # Force derivative to keep it at zero
                dpdt[N-1] = 0
            else:
                # Reflecting BC: same no-flux condition as left
                if D > 1e-10:
                    p_ghost_right = p[N-2] + (4*dx*A_const/D)*p[N-1]
                    dpdt[N-1] = (-A_const * (p_ghost_right - p[N-2]) / (2*dx) +
                                 (D/2) * (p_ghost_right - 2*p[N-1] + p[N-2]) / (dx**2))
                else:
                    dpdt[N-1] = -A_const * (p[N-1] - p[N-2]) / dx

        elif drift_type == 'harmonic':
            # Case B: dp/dt = d/dx(a*x*p) + (D/2) * d²p/dx²

            # --- INTERIOR POINTS ---
            # Advection term: d/dx(a*x*p) discretized with central difference
            flux_advection = a_harm * x[1:-1] * p[1:-1]
            flux_advection_plus = a_harm * x[2:] * p[2:]
            flux_advection_minus = a_harm * x[:-2] * p[:-2]

            dpdt[1:-1] = ((flux_advection_plus - flux_advection_minus) / (2*dx) +
                          (D/2) * (p[2:] - 2*p[1:-1] + p[:-2]) / (dx**2))

            # --- BOUNDARIES: NO-FLUX (REFLECTING) ---
            # Flux J = -a*x*p + (D/2)*dp/dx = 0
            # => dp/dx = (2*a*x/D)*p

            if D > 1e-10:
                # Left boundary (i=0)
                p_ghost_left = p[1] - (4*dx*a_harm*x[0]/D)*p[0]
                x_ghost_left = x[0] - dx

                flux_adv_0 = a_harm * x[0] * p[0]
                flux_adv_ghost = a_harm * x_ghost_left * p_ghost_left
                flux_adv_1 = a_harm * x[1] * p[1]

                dpdt[0] = ((flux_adv_1 - flux_adv_ghost) / (2*dx) +
                           (D/2) * (p[1] - 2*p[0] + p_ghost_left) / (dx**2))

                # Right boundary (i=N-1)
                p_ghost_right = p[N-2] + (4*dx*a_harm*x[N-1]/D)*p[N-1]
                x_ghost_right = x[N-1] + dx

                flux_adv_N = a_harm * x[N-1] * p[N-1]
                flux_adv_Nm1 = a_harm * x[N-2] * p[N-2]
                flux_adv_ghost_r = a_harm * x_ghost_right * p_ghost_right

                dpdt[N-1] = ((flux_adv_ghost_r - flux_adv_Nm1) / (2*dx) +
                             (D/2) * (p_ghost_right - 2*p[N-1] + p[N-2]) / (dx**2))

        return dpdt

    return (fp_rhs,)


@app.cell
def _(fp_rhs, get_initial_condition, np, solve_ivp):
    """Main solver function"""
    def solve_fp_equation(N, L, T_max, D, drift_type, A_const=0, a_harm=0, 
                          l_width=1.0, boundary_right='reflecting', n_snapshots=100):
        """
        Solve the Fokker-Planck equation using Method of Lines.

        Args:
            N: number of spatial grid points
            L: domain length (from -L/2 to +L/2)
            T_max: final time for simulation
            D: diffusion coefficient
            drift_type: 'constant' or 'harmonic'
            A_const: constant drift velocity
            a_harm: harmonic trap strength
            l_width: width of initial pulse
            boundary_right: 'reflecting' or 'absorbing'
            n_snapshots: number of time snapshots to save

        Returns:
            sol: solution object from solve_ivp
            x: spatial grid
            dx: grid spacing
        """
        # Create spatial grid
        x = np.linspace(-L/2, L/2, N)
        dx = L / (N - 1)

        # Generate initial condition
        p0 = get_initial_condition(x, l_width)

        # Ensure consistency with absorbing boundary
        if boundary_right == 'absorbing':
            p0[-1] = 0.0

        # Time points for solution output
        t_eval = np.linspace(0, T_max, n_snapshots)

        # Solve the ODE system using RK45 (4th/5th order Runge-Kutta)
        sol = solve_ivp(
            fun=fp_rhs,
            t_span=(0, T_max),
            y0=p0,
            t_eval=t_eval,
            args=(x, dx, D, drift_type, A_const, a_harm, boundary_right),
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )

        return sol, x, dx

    return (solve_fp_equation,)


@app.cell
def _(mo):
    """UI Controls - General Parameters"""
    mo.md("## Simulation Parameters")
    return


@app.cell
def _(mo):
    """General parameter sliders"""
    L_slider = mo.ui.slider(5, 20, value=10, label="Domain Length (L)", step=1)
    N_slider = mo.ui.slider(50, 500, value=100, label="Grid Points (N)", step=10)
    D_slider = mo.ui.number(0, 5, value=1.0, step=0.1, label="Diffusion Coefficient (D)")
    T_slider = mo.ui.slider(1, 50, value=10, label="Maximum Time (T)", step=1)

    return D_slider, L_slider, N_slider, T_slider


@app.cell
def _(mo):
    """Question A/C specific controls"""
    A_slider = mo.ui.number(-2, 2, value=0.0, step=0.1, label="Drift Velocity (A)")
    l_width_slider = mo.ui.number(0.1, 10, value=1.0, step=0.1, label="Pulse Width (ℓ)")

    return A_slider, l_width_slider


@app.cell
def _(mo):
    """Question B specific controls"""
    a_harm_slider = mo.ui.number(0.1, 2.0, value=0.5, step=0.1, label="Trap Strength (a)")
    l_harm_width = mo.ui.number(0.1, 5, value=2.0, step=0.1, label="Initial Width (ℓ)")

    return a_harm_slider, l_harm_width


@app.cell
def _(np, plt, solve_fp_equation):
    """Question A1: Early and Long-time evolution plots"""
    def plot_A1_evolution(N, L, T_max, D, A, l_width):
        """Plot early-time and long-time evolution for Question A1"""
        sol, x, dx = solve_fp_equation(
            N=N, L=L, T_max=T_max, D=D, 
            drift_type='constant', A_const=A, 
            l_width=l_width, boundary_right='reflecting'
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Early-time behavior (intervals of 0.2 in time)
        n_early = min(len(sol.t), 50)
        time_early = np.linspace(0, n_early-1, min(10, n_early), dtype=int)

        for idx in time_early:
            if idx < len(sol.t):
                ax1.plot(x, sol.y[:, idx], label=f't={sol.t[idx]:.2f}')

        ax1.set_xlabel('x')
        ax1.set_ylabel('p(x,t)')
        ax1.set_title('Early-Time Evolution (A = {:.1f})'.format(A))
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Long-time behavior (larger intervals)
        time_late = np.linspace(0, len(sol.t)-1, 10, dtype=int)

        cmap = plt.cm.viridis
        for i, idx in enumerate(time_late):
            color = cmap(i / len(time_late))
            ax2.plot(x, sol.y[:, idx], color=color, label=f't={sol.t[idx]:.1f}')

        ax2.set_xlabel('x')
        ax2.set_ylabel('p(x,t)')
        ax2.set_title('Long-Time Evolution')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    return (plot_A1_evolution,)


@app.cell
def _(np, plt, solve_fp_equation):
    """Question A2: Gaussian fit analysis"""
    def plot_A2_gaussian_fit(N, L, T_max, D, A, l_width):
        """Plot Gaussian fits at different times for Question A2"""
        sol, x, dx = solve_fp_equation(
            N=N, L=L, T_max=T_max, D=D,
            drift_type='constant', A_const=A,
            l_width=l_width, boundary_right='reflecting'
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        # Select 6 different times to show Gaussian convergence
        time_indices = np.linspace(0, len(sol.t)-1, 6, dtype=int)

        for i, idx in enumerate(time_indices):
            p_current = sol.y[:, idx]

            # Calculate mean and variance
            mean_x = np.trapezoid(x * p_current, x)
            variance = np.trapezoid((x - mean_x)**2 * p_current, x)
            sigma = np.sqrt(variance)

            # Generate Gaussian with same mean and variance
            if sigma > 1e-10:
                gaussian = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean_x) / sigma)**2)
            else:
                gaussian = np.zeros_like(x)

            # Plot
            axes[i].plot(x, p_current, 'b-', linewidth=2, label='Simulation')
            axes[i].plot(x, gaussian, 'r--', linewidth=2, label='Gaussian Fit')
            axes[i].set_title(f't = {sol.t[idx]:.2f}, σ² = {variance:.3f}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('p(x,t)')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    return (plot_A2_gaussian_fit,)


@app.cell
def _(np, plt, solve_fp_equation):
    """Question A4: Comparison of different initial conditions"""
    def plot_A4_comparison(N, L, T_max, D, A):
        """Compare evolution for ℓ=1 vs ℓ=L (uniform) for Question A4"""
        # Simulation 1: ℓ = 1
        sol1, x, dx = solve_fp_equation(
            N=N, L=L, T_max=T_max, D=D,
            drift_type='constant', A_const=A,
            l_width=1.0, boundary_right='reflecting'
        )

        # Simulation 2: ℓ = L (uniform initial distribution)
        sol2, x, dx = solve_fp_equation(
            N=N, L=L, T_max=T_max, D=D,
            drift_type='constant', A_const=A,
            l_width=L, boundary_right='reflecting'
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Evolution comparison
        time_indices = np.linspace(0, len(sol1.t)-1, 8, dtype=int)

        for idx in time_indices:
            alpha = 0.3 + 0.7 * (idx / len(time_indices))
            axes[0].plot(x, sol1.y[:, idx], 'b-', alpha= 0.5, linewidth=1.5)
            axes[1].plot(x, sol2.y[:, idx], 'r-', alpha= 0.5, linewidth=1.5)

        axes[0].set_title(f'ℓ = 1 (Pulse IC), A = {A}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('p(x,t)')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f'ℓ = L (Uniform IC), A = {A}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('p(x,t)')
        axes[1].grid(True, alpha=0.3)

        # Add final state comparison
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, sol1.y[:, -1], 'b-', linewidth=2, label='ℓ = 1 (final state)')
        ax.plot(x, sol2.y[:, -1], 'r--', linewidth=2, label='ℓ = L (final state)')
        ax.set_xlabel('x')
        ax.set_ylabel('p(x,t)')
        ax.set_title(f'Final State Comparison (A = {A}, D = {D})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, fig2

    return (plot_A4_comparison,)


@app.cell
def _(np, plt, solve_fp_equation):
    """Question B: Harmonic potential evolution"""
    def plot_B_harmonic(N, L, T_max, D, a_values, l_width):
        """Plot evolution for different trap strengths (Question B)"""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, a in enumerate(a_values):
            sol, x, dx = solve_fp_equation(
                N=N, L=L, T_max=T_max, D=D,
                drift_type='harmonic', a_harm=a,
                l_width=l_width, boundary_right='reflecting'
            )

            # Plot evolution
            time_indices = np.linspace(0, len(sol.t)-1, 10, dtype=int)
            cmap = plt.cm.plasma

            for j, idx in enumerate(time_indices):
                color = cmap(j / len(time_indices))
                axes[i].plot(x, sol.y[:, idx], color=color, alpha=0.7)

            axes[i].set_title(f'a = {a}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('p(x,t)')
            axes[i].grid(True, alpha=0.3)

        # Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                    norm=plt.Normalize(vmin=0, vmax=T_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                           pad=0.1, aspect=30)
        cbar.set_label('Time')

        plt.suptitle(f'Harmonic Potential Evolution (D = {D}, ℓ = {l_width})', 
                     fontsize=14, y=1.02)
        #plt.tight_layout()

        return fig

    return (plot_B_harmonic,)


@app.cell
def _(np, plt, solve_fp_equation):
    """Question C: Absorbing boundary - probability decay"""
    def plot_C_absorbing(N, L, T_max, D, A_values):
        """Plot probability decay for absorbing boundary (Question C)"""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ['blue', 'green', 'red']

        for A_val, color in zip(A_values, colors):
            sol, x, dx = solve_fp_equation(
                N=N, L=L, T_max=T_max, D=D,
                drift_type='constant', A_const=A_val,
                l_width=1.0, boundary_right='absorbing'
            )

            # Calculate total probability over time
            P_t = np.trapezoid(sol.y, x, axis=0)

            # Linear plot
            axes[0].plot(sol.t, P_t, color=color, linewidth=2.5, 
                        label=f'A = {A_val}')

            # Log-linear plot
            mask = P_t > 1e-10
            axes[1].semilogy(sol.t[mask], P_t[mask], color=color, 
                            linewidth=2.5, label=f'A = {A_val}')

            # Log-log plot
            t_mask = (sol.t > 1e-3) & mask
            if np.sum(t_mask) > 0:
                axes[2].loglog(sol.t[t_mask], P_t[t_mask], color=color, 
                              linewidth=2.5, label=f'A = {A_val}')

        # Formatting
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('P(t) = ∫p(x,t)dx')
        axes[0].set_title('Linear Scale')
        axes[0].set_ylim(0, 1.05)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('log[P(t)]')
        axes[1].set_title('Semi-Log Scale')
        axes[1].legend()
        axes[1].grid(True, which='both', alpha=0.3)

        axes[2].set_xlabel('log(Time)')
        axes[2].set_ylabel('log[P(t)]')
        axes[2].set_title('Log-Log Scale')
        axes[2].legend()
        axes[2].grid(True, which='both', alpha=0.3)

        plt.suptitle(f'Probability Decay with Absorbing Boundary (D = {D}, ℓ = 1)', 
                     fontsize=14, y=1.0)
        plt.tight_layout()

        return fig

    return (plot_C_absorbing,)


@app.cell
def _(
    A_slider,
    D_slider,
    L_slider,
    N_slider,
    T_slider,
    l_width_slider,
    mo,
    plot_A1_evolution,
    plot_A2_gaussian_fit,
):
    """Question A Section - Interactive Interface"""

    mo.md("## Question A: Constant Drift with Reflecting Boundaries")

    mo.md(
        r"""
        **Task A1:** Set A=0, ℓ=1. Observe spreading behavior.  
        **Task A2:** Check Gaussian fit convergence.  
        **Task A3:** Comment on steady state.  
        **Task A4:** Set A=1, compare ℓ=1 vs ℓ=L.
        """
    )

    # Controls
    controls_A = mo.hstack([
        mo.vstack([L_slider, N_slider]),
        mo.vstack([D_slider, T_slider]),
        mo.vstack([A_slider, l_width_slider])
    ], justify="start")

    # Generate plots
    fig_A1 = plot_A1_evolution(
        N_slider.value, L_slider.value, T_slider.value,
        D_slider.value, A_slider.value, l_width_slider.value
    )

    fig_A2 = plot_A2_gaussian_fit(
        N_slider.value, L_slider.value, T_slider.value,
        D_slider.value, A_slider.value, l_width_slider.value
    )

    mo.vstack([controls_A, fig_A1, fig_A2])

    return


@app.cell
def _(
    A_slider,
    D_slider,
    L_slider,
    N_slider,
    T_slider,
    mo,
    plot_A4_comparison,
):
    """Question A4 - Initial Condition Comparison"""

    mo.md("### A4: Initial Condition Comparison (ℓ=1 vs ℓ=L)")

    fig_A4a, fig_A4b = plot_A4_comparison(
        N_slider.value, L_slider.value, T_slider.value,
        D_slider.value, A_slider.value
    )

    mo.vstack([fig_A4a, fig_A4b])

    return


@app.cell
def _(
    D_slider,
    L_slider,
    N_slider,
    T_slider,
    a_harm_slider,
    l_harm_width,
    mo,
    plot_B_harmonic,
):
    """Question B Section - Harmonic Potential"""

    mo.md("## Question B: Harmonic Potential (Optical Trap)")

    mo.md(
        r"""
        The particle experiences a restoring force: A(x) = -ax

        Simulate for **a = 0.2, 0.5, 1.0** to observe confinement effects.
        """
    )

    controls_B = mo.hstack([
        mo.vstack([L_slider, N_slider]),
        mo.vstack([D_slider, T_slider]),
        mo.vstack([a_harm_slider, l_harm_width])
    ], justify="start")

    # Generate plot for multiple a values
    fig_B = plot_B_harmonic(
        N_slider.value, L_slider.value, T_slider.value,
        D_slider.value, [0.2, 0.5, 1.0], l_harm_width.value
    )

    mo.vstack([controls_B, fig_B])

    return


@app.cell
def _(D_slider, L_slider, N_slider, T_slider, mo, plot_C_absorbing):
    """Question C Section - Absorbing Boundary"""

    mo.md("## Question C: Absorbing Boundary at Right End")

    mo.md(
        r"""
        Right boundary condition: **p(L/2, t) = 0** (particles absorbed)

        Compare **A = 0, 0.2, 1.0** to analyze escape probability.

        Fixed: ℓ = 1, D as set above
        """
    )

    controls_C = mo.hstack([
        mo.vstack([L_slider, N_slider]),
        mo.vstack([D_slider, T_slider])
    ], justify="start")

    # Generate plot for A = 0, 0.2, 1.0
    fig_C = plot_C_absorbing(
        N_slider.value, L_slider.value, T_slider.value,
        D_slider.value, [0, 0.2, 1.0]
    )

    mo.vstack([controls_C, fig_C])

    return


@app.cell
def _(mo):
    """Discussion and Analysis Section"""
    mo.md(
        r"""
        ---

        ## Analysis and Discussion

        ### Question A Analysis

        **A1 & A2: Spreading and Gaussian Convergence**
        - Initial pulse spreads due to diffusion
        - At early times, the square pulse shape is evident
        - As time progresses, the distribution becomes increasingly Gaussian
        - Variance grows linearly with time: σ²(t) ≈ D·t (for A=0)

        **A3: Steady State with Reflecting Boundaries**
        - For A=0: Uniform distribution p = 1/L (equal probability everywhere)
        - Physical interpretation: Particle has equal likelihood to be anywhere
        - Could predict from symmetry and no-flux boundaries

        **A4: Effect of Drift**
        - With A ≠ 0: Final state is NOT uniform
        - Probability accumulates at downstream boundary
        - Initial condition matters less at long times (ergodicity)
        - Difference from pure advection (D=0): particles don't all move together

        ### Question B Analysis

        **Harmonic Confinement**
        - Larger 'a' → stronger confinement → narrower steady-state distribution
        - Steady state is a balance between drift toward origin and diffusion
        - Final distribution: Boltzmann-like, p(x) ∝ exp(-ax²/D)
        - Unlike Question A: particles are trapped, not uniformly distributed
        - Drag coefficient ∝ 1/a: smaller drag → faster equilibration

        ### Question C Analysis

        **Probability Decay with Absorption**
        - A = 0: Symmetric diffusion, slower escape
        - A > 0: Drift accelerates particle loss
        - Decay is NOT linear (would be straight line on linear plot)
        - Semi-log plot: Exponential decay at long times
        - Log-log plot: May show power-law at early times

        **Effect of Diffusion vs Pure Advection**
        - D = 0, A > 0: All particles reach boundary at t ≈ L/(2A)
        - D > 0: Diffusion causes spread → particles escape over range of times
        - Diffusion allows "upstream" escape even with positive drift

        ---

        ## Numerical Implementation Notes

        1. **Method of Lines**: Spatial discretization → system of ODEs
        2. **Central Differences**: 2nd order accurate in space
        3. **Ghost Nodes**: Elegant way to implement boundary conditions
        4. **CFL Condition**: Automatically handled by adaptive RK45
        5. **Normalization**: ∫p(x,t)dx should remain 1 (except for absorbing BC)

        """
    )
    return


if __name__ == "__main__":
    app.run()
