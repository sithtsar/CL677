import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium", layout_file="layouts/notebook.slides.json")


@app.cell
def _():
    from io import BytesIO

    from pathlib import Path

    import marimo as mo

    import matplotlib.image as mpimg

    import matplotlib.pyplot as plt

    import numpy as np

    from scipy.integrate import cumulative_trapezoid

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams["figure.dpi"] = 140

    plt.rcParams["axes.spines.top"] = False

    plt.rcParams["axes.spines.right"] = False
    return BytesIO, Path, cumulative_trapezoid, mo, mpimg, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Project A: Smooth Random Noise & the Stratonovich Limit

    ## Overview

    In this project, we study how **smooth random functions** can be used to model stochastic forcing in differential equations, and how the classical **Stratonovich limit** emerges as the correlation time of the noise is reduced.

    Rather than working directly with white noise, we construct a smooth random signal and treat the resulting system like an ordinary differential equation. This makes it possible to use standard numerical solvers while still capturing the essential stochastic behavior.

    ## Reference

    This project is based on the ideas presented in:

    - Filip, Javeed, and Trefethen, *SIAM Review* **61**(1), 185-2017
    - `docs/random.pdf`

    ## Goals

    The main objectives of this project are:

    1. **Construct smooth random functions** from scratch, without using ready-made Chebfun tools.
    2. **Use these functions to model Brownian motion** and reproduce the key figures from the reference paper.
    3. **Apply smooth random forcing to the geometric random walk problem** from Assignment 3.
    4. **Verify the white-noise limit** by showing that, as the correlation time tends to zero, the solution approaches the **Stratonovich** interpretation rather than the **Itô** interpretation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem Statement

    The course brief asks for a single notebook that does two jobs well:

    - reproduce the paper's Figure 1 and Figure 6 using a from-scratch Fourier construction of smooth random functions
    - connect those constructions to Assignment 3 by showing that the geometric random walk driven by smooth random forcing approaches the **Stratonovich** model

    This version keeps the notebook interactive: the main figures and diagnostics are controlled by marimo UI elements, and matplotlib outputs are rendered through marimo's interactive viewer.
    """)
    return


@app.cell
def _(BytesIO, Path, cumulative_trapezoid, mpimg, np):
    helper_repo_root = Path.cwd()

    ground_truth_dir = helper_repo_root / "docs" / "ground_truth"


    def centered_coordinate_grid(x, interval):
        interval_left, interval_right = interval
        return x - 0.5 * (
            interval_left + interval_right
        ), interval_right - interval_left


    def smooth_random_function(
        x,
        wavelength,
        interval=(-1.0, 1.0),
        normalization="standard",
        seed=0,
    ):
        sample_x = np.asarray(x, dtype=float)
        shifted_x, domain_length = centered_coordinate_grid(sample_x, interval)
        truncation_degree = max(1, int(np.floor(domain_length / wavelength)))
        coeff_std = np.sqrt(1.0 / (2 * truncation_degree + 1))
        if normalization == "big":
            coeff_std *= np.sqrt(2.0 / wavelength)
        elif normalization != "standard":
            raise ValueError(f"unknown normalization: {normalization}")

        coeff_rng = np.random.default_rng(seed)
        cosine_coeffs = coeff_rng.normal(
            0.0, coeff_std, size=truncation_degree + 1
        )
        sine_coeffs = coeff_rng.normal(0.0, coeff_std, size=truncation_degree)
        harmonic_ids = np.arange(1, truncation_degree + 1)[:, None]
        harmonic_phase = (
            2.0 * np.pi * harmonic_ids * shifted_x[None, :] / domain_length
        )
        return cosine_coeffs[0] + np.sqrt(2.0) * (
            (cosine_coeffs[1:, None] * np.cos(harmonic_phase)).sum(axis=0)
            + (sine_coeffs[:, None] * np.sin(harmonic_phase)).sum(axis=0)
        )


    def chebfun_like_big_random_function(
        x,
        wavelength,
        interval=(0.0, 1.0),
        seed=0,
    ):
        sample_x = np.asarray(x, dtype=float)
        shifted_x, domain_length = centered_coordinate_grid(sample_x, interval)
        truncation_degree = max(1, int(np.floor(domain_length / wavelength)))
        coeff_std = np.sqrt(1.0 / (2 * truncation_degree + 1))
        coeff_rng = np.random.default_rng(seed)
        complex_coeffs = {
            0: coeff_rng.normal(0.0, coeff_std)
            + 1j * coeff_rng.normal(0.0, coeff_std)
        }
        for harmonic_id in range(1, truncation_degree + 1):
            complex_coeffs[harmonic_id] = coeff_rng.normal(
                0.0, coeff_std
            ) + 1j * coeff_rng.normal(0.0, coeff_std)
            complex_coeffs[-harmonic_id] = coeff_rng.normal(
                0.0, coeff_std
            ) + 1j * coeff_rng.normal(0.0, coeff_std)

        complex_series = np.zeros_like(sample_x, dtype=complex)
        for harmonic_id in range(-truncation_degree, truncation_degree + 1):
            complex_series += complex_coeffs[harmonic_id] * np.exp(
                2j * np.pi * harmonic_id * shifted_x / domain_length
            )
        return np.real(complex_series) * np.sqrt(2.0 / wavelength)


    def nested_smooth_random_function(
        x,
        wavelength,
        interval,
        base_cosine_coeffs,
        base_sine_coeffs,
        normalization="standard",
    ):
        sample_x = np.asarray(x, dtype=float)
        shifted_x, domain_length = centered_coordinate_grid(sample_x, interval)
        truncation_degree = max(1, int(np.floor(domain_length / wavelength)))
        coeff_std = np.sqrt(1.0 / (2 * truncation_degree + 1))
        if normalization == "big":
            coeff_std *= np.sqrt(2.0 / wavelength)
        elif normalization != "standard":
            raise ValueError(f"unknown normalization: {normalization}")

        cosine_coeffs = coeff_std * base_cosine_coeffs[: truncation_degree + 1]
        sine_coeffs = coeff_std * base_sine_coeffs[:truncation_degree]
        harmonic_ids = np.arange(1, truncation_degree + 1)[:, None]
        harmonic_phase = (
            2.0 * np.pi * harmonic_ids * shifted_x[None, :] / domain_length
        )
        return cosine_coeffs[0] + np.sqrt(2.0) * (
            (cosine_coeffs[1:, None] * np.cos(harmonic_phase)).sum(axis=0)
            + (sine_coeffs[:, None] * np.sin(harmonic_phase)).sum(axis=0)
        )


    def integrate_indefinite(x, values):
        return cumulative_trapezoid(values, x, initial=0.0)


    def figure_to_rgba_array(figure_object):
        image_buffer = BytesIO()
        figure_object.savefig(image_buffer, format="png", bbox_inches="tight")
        image_buffer.seek(0)
        return mpimg.imread(image_buffer)


    def simulate_sde_ensembles(
        num_paths=3000, t_end=1.5, dt=1e-3, x0=1.0, seed=2026
    ):
        time_grid = np.linspace(0.0, t_end, int(round(t_end / dt)) + 1)
        step_noise_rng = np.random.default_rng(seed)
        brownian_steps = np.sqrt(dt) * step_noise_rng.normal(
            size=(num_paths, time_grid.size - 1)
        )

        ito_paths = np.empty((num_paths, time_grid.size))
        strat_paths = np.empty((num_paths, time_grid.size))
        ito_paths[:, 0] = x0
        strat_paths[:, 0] = x0

        for step_id in range(time_grid.size - 1):
            ito_paths[:, step_id + 1] = ito_paths[:, step_id] * (
                1.0 + brownian_steps[:, step_id]
            )
            strat_paths[:, step_id + 1] = strat_paths[:, step_id] * (
                1.0 + 0.5 * dt + brownian_steps[:, step_id]
            )

        return time_grid, ito_paths, strat_paths


    def simulate_smooth_ode_ensembles(
        lambda_values,
        num_paths=1000,
        t_end=1.5,
        num_points=1501,
        x0=1.0,
        seed=300,
    ):
        ode_time_grid = np.linspace(0.0, t_end, num_points)
        max_degree = int(np.floor(t_end / min(lambda_values))) + 2
        nested_rng = np.random.default_rng(seed)
        base_cosine_coeffs = nested_rng.normal(size=(num_paths, max_degree + 1))
        base_sine_coeffs = nested_rng.normal(size=(num_paths, max_degree))

        ode_path_dict = {}
        for lambda_value in lambda_values:
            lambda_paths = np.empty((num_paths, num_points))
            for path_id in range(num_paths):
                lambda_forcing = nested_smooth_random_function(
                    ode_time_grid,
                    wavelength=lambda_value,
                    interval=(0.0, t_end),
                    base_cosine_coeffs=base_cosine_coeffs[path_id],
                    base_sine_coeffs=base_sine_coeffs[path_id],
                    normalization="big",
                )
                lambda_walk = integrate_indefinite(ode_time_grid, lambda_forcing)
                lambda_paths[path_id] = x0 * np.exp(lambda_walk)
            ode_path_dict[lambda_value] = lambda_paths
        return ode_time_grid, ode_path_dict

    return (
        chebfun_like_big_random_function,
        figure_to_rgba_array,
        ground_truth_dir,
        integrate_indefinite,
        nested_smooth_random_function,
        simulate_sde_ensembles,
        simulate_smooth_ode_ensembles,
        smooth_random_function,
    )


@app.cell
def _(mo):
    figure1_lambda_ui = mo.ui.slider(
        steps=[0.1, 0.025],
        value=0.1,
        show_value=True,
        label="Figure 1 wavelength",
    )

    figure1_norm_ui = mo.ui.radio(
        options=["standard", "big"],
        value="standard",
        label="Normalization",
    )

    figure6_lambda_ui = mo.ui.slider(
        steps=[1 / 5, 1 / 25, 1 / 125],
        value=1 / 25,
        show_value=True,
        label="Figure 6 wavelength",
    )

    brownian_lambda_ui = mo.ui.slider(
        steps=[0.2, 0.1, 0.05, 0.025],
        value=0.05,
        show_value=True,
        label="Brownian diagnostic lambda",
    )

    moment_metric_ui = mo.ui.radio(
        options=["mean", "second"], value="mean", label="Moment to compare"
    )

    moment_lambda_ui = mo.ui.slider(
        steps=[0.25, 0.1, 0.05, 0.025],
        value=0.025,
        show_value=True,
        label="Smooth-ODE lambda",
    )

    show_empirical_ui = mo.ui.switch(
        value=True, label="Show simulated ensemble curves"
    )

    snapshot_time_ui = mo.ui.slider(
        steps=[0.5, 1.0, 1.5],
        value=1.0,
        show_value=True,
        label="Distribution snapshot time",
    )
    return (
        brownian_lambda_ui,
        figure1_lambda_ui,
        figure1_norm_ui,
        figure6_lambda_ui,
        moment_lambda_ui,
        moment_metric_ui,
        show_empirical_ui,
        snapshot_time_ui,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mathematical Setup

    For an interval of length $L$ and wavelength parameter $\lambda$, we use the truncated Fourier series

    \[
    f(x) = a_0 + \sqrt{2}\sum_{j=1}^{m}
    \left[
    a_j \cos\left(\frac{2\pi jx}{L}\right)
    +
    b_j \sin\left(\frac{2\pi jx}{L}\right)
    \right],
    \qquad
    m = \lfloor L/\lambda \rfloor.
    \]

    The coefficient distributions are

    \[
    a_j, b_j \sim N\!\left(0, \frac{1}{2m+1}\right)
    \]

    for the standard normalization, and the whole function is multiplied by

    \[
    \sqrt{\frac{2}{\lambda}}
    \]

    for the big normalization. The standard normalization is appropriate when function values are the main object. The big normalization is appropriate when the **integral** of the forcing is the main object, because it produces Brownian scaling in the limit $\lambda \to 0$.
    """)
    return


@app.cell
def _(ground_truth_dir, np, smooth_random_function):
    figure1_x_grid = np.linspace(-1.0, 1.0, 2000)

    figure1_truth_paths = {
        (0.1, "standard"): ground_truth_dir / "figure1" / "panel_a.jpeg",
        (0.1, "big"): ground_truth_dir / "figure1" / "panel_c.jpeg",
        (0.025, "standard"): ground_truth_dir / "figure1" / "panel_b.jpeg",
        (0.025, "big"): ground_truth_dir / "figure1" / "panel_d.jpeg",
    }

    figure1_generated_samples = {
        (0.1, "standard"): smooth_random_function(
            figure1_x_grid,
            wavelength=0.1,
            interval=(-1.0, 1.0),
            normalization="standard",
            seed=7,
        ),
        (0.1, "big"): smooth_random_function(
            figure1_x_grid,
            wavelength=0.1,
            interval=(-1.0, 1.0),
            normalization="big",
            seed=7,
        ),
        (0.025, "standard"): smooth_random_function(
            figure1_x_grid,
            wavelength=0.025,
            interval=(-1.0, 1.0),
            normalization="standard",
            seed=17,
        ),
        (0.025, "big"): smooth_random_function(
            figure1_x_grid,
            wavelength=0.025,
            interval=(-1.0, 1.0),
            normalization="big",
            seed=17,
        ),
    }
    return figure1_generated_samples, figure1_truth_paths, figure1_x_grid


@app.cell(hide_code=True)
def _(
    figure1_generated_samples,
    figure1_lambda_ui,
    figure1_norm_ui,
    figure1_truth_paths,
    figure1_x_grid,
    figure_to_rgba_array,
    mo,
    np,
    plt,
):
    figure1_selected_key = (figure1_lambda_ui.value, figure1_norm_ui.value)

    figure1_selected_sample = figure1_generated_samples[figure1_selected_key]

    figure1_selected_truth = figure1_truth_paths[figure1_selected_key]

    figure1_plot_figure, figure1_plot_axes = plt.subplots(figsize=(5.8, 3.6))

    figure1_plot_axes.plot(
        figure1_x_grid, figure1_selected_sample, color="#0b5d8f", lw=1.5
    )

    figure1_plot_axes.set_xlim(-1.0, 1.0)

    figure1_plot_axes.set_xlabel("x")

    figure1_plot_axes.set_ylabel("f(x)")

    figure1_plot_axes.set_title(
        f"Reproduced Figure 1 panel: lambda={figure1_lambda_ui.value}, normalization={figure1_norm_ui.value}"
    )

    figure1_plot_axes.set_ylim(
        1.1 * np.min(figure1_selected_sample),
        1.1 * np.max(figure1_selected_sample),
    )

    figure1_plot_image = figure_to_rgba_array(figure1_plot_figure)

    plt.close(figure1_plot_figure)

    mo.vstack(
        [
            mo.md(
                f"""
                ## Figure 1 Reproduction

                Use the controls below to switch between the two wavelengths and the two normalizations discussed in the paper.
                """
            ),
            mo.hstack([figure1_lambda_ui, figure1_norm_ui], justify="start"),
            mo.hstack(
                [
                    mo.image(
                        figure1_selected_truth,
                        width="49%",
                        height=420,
                        caption="Paper panel",
                        style={"object-fit": "contain"},
                    ),
                    mo.image(
                        figure1_plot_image,
                        width="49%",
                        height=420,
                        caption="Notebook reproduction",
                        style={"object-fit": "contain"},
                    ),
                ],
                justify="space-between",
            ),
        ]
    )
    return


@app.cell
def _(integrate_indefinite, nested_smooth_random_function, np):
    variance_time_grid = np.linspace(0.0, 1.0, 1201)

    variance_lambda_values = [0.2, 0.1, 0.05, 0.025]

    variance_num_paths = 300

    variance_max_degree = int(np.floor(1.0 / min(variance_lambda_values))) + 3

    variance_rng = np.random.default_rng(1234)

    variance_base_cosines = variance_rng.normal(
        size=(variance_num_paths, variance_max_degree + 1)
    )

    variance_base_sines = variance_rng.normal(
        size=(variance_num_paths, variance_max_degree)
    )

    variance_standard_samples = []

    variance_brownian_curves = {}

    for variance_path_id in range(variance_num_paths):
        variance_standard_samples.append(
            nested_smooth_random_function(
                variance_time_grid,
                wavelength=0.05,
                interval=(0.0, 1.0),
                base_cosine_coeffs=variance_base_cosines[variance_path_id],
                base_sine_coeffs=variance_base_sines[variance_path_id],
                normalization="standard",
            )
        )

    standard_point_variance = float(
        np.var(np.array(variance_standard_samples)[:, 500])
    )

    for variance_lambda_value in variance_lambda_values:
        variance_walk_stack = []
        for variance_path_id in range(variance_num_paths):
            variance_lambda_forcing = nested_smooth_random_function(
                variance_time_grid,
                wavelength=variance_lambda_value,
                interval=(0.0, 1.0),
                base_cosine_coeffs=variance_base_cosines[variance_path_id],
                base_sine_coeffs=variance_base_sines[variance_path_id],
                normalization="big",
            )
            variance_walk_stack.append(
                integrate_indefinite(variance_time_grid, variance_lambda_forcing)
            )
        variance_brownian_curves[variance_lambda_value] = np.array(
            variance_walk_stack
        ).var(axis=0)
    return (
        standard_point_variance,
        variance_brownian_curves,
        variance_time_grid,
    )


@app.cell(hide_code=True)
def _(
    brownian_lambda_ui,
    mo,
    plt,
    standard_point_variance,
    variance_brownian_curves,
    variance_time_grid,
):
    brownian_plot_figure, brownian_plot_axes = plt.subplots(figsize=(6.5, 4.0))

    brownian_plot_axes.plot(
        variance_time_grid,
        variance_time_grid,
        color="black",
        lw=2,
        ls="--",
        label="Brownian target",
    )

    brownian_plot_axes.plot(
        variance_time_grid,
        variance_brownian_curves[brownian_lambda_ui.value],
        color="#006d77",
        lw=2.2,
        label=rf"big normalization, $\lambda={brownian_lambda_ui.value}$",
    )

    brownian_plot_axes.set_xlabel("t")

    brownian_plot_axes.set_ylabel("ensemble variance")

    brownian_plot_axes.set_title(
        f"Integrated big smooth random functions; pointwise standard variance = {standard_point_variance:.3f}"
    )

    brownian_plot_axes.legend()

    mo.vstack(
        [
            mo.md(
                r"""
                ## Brownian Convergence Diagnostic

                The slider chooses which $\lambda$ value to compare against the Brownian variance law

                \[
                \mathrm{Var}[W(t)] = t.
                \]
                """
            ),
            brownian_lambda_ui,
            mo.mpl.interactive(brownian_plot_figure),
        ]
    )
    return


@app.cell
def _(
    chebfun_like_big_random_function,
    ground_truth_dir,
    integrate_indefinite,
    np,
):
    figure6_time_grid = np.linspace(0.0, 1.0, 2400)

    figure6_truth_paths = {
        1 / 5: ground_truth_dir / "figure6" / "panel_a.jpeg",
        1 / 25: ground_truth_dir / "figure6" / "panel_b.jpeg",
        1 / 125: ground_truth_dir / "figure6" / "panel_c.jpeg",
    }

    figure6_lambda_values = [1 / 5, 1 / 25, 1 / 125]

    figure6_seed = 19

    figure6_walks = {}

    for figure6_lambda_value in figure6_lambda_values:
        figure6_lambda_forcing = chebfun_like_big_random_function(
            figure6_time_grid,
            wavelength=figure6_lambda_value,
            interval=(0.0, 1.0),
            seed=figure6_seed,
        )
        figure6_walks[figure6_lambda_value] = integrate_indefinite(
            figure6_time_grid, figure6_lambda_forcing
        )
    return figure6_seed, figure6_time_grid, figure6_truth_paths, figure6_walks


@app.cell(hide_code=True)
def _(
    figure6_lambda_ui,
    figure6_seed,
    figure6_time_grid,
    figure6_truth_paths,
    figure6_walks,
    figure_to_rgba_array,
    mo,
    plt,
):
    figure6_selected_truth = figure6_truth_paths[figure6_lambda_ui.value]

    figure6_selected_walk = figure6_walks[figure6_lambda_ui.value]

    figure6_plot_figure, figure6_plot_axes = plt.subplots(figsize=(5.8, 3.6))

    figure6_plot_axes.plot(
        figure6_time_grid, figure6_selected_walk, color="#006d77", lw=1.7
    )

    figure6_plot_axes.set_xlim(0.0, 1.0)

    figure6_plot_axes.set_xlabel("t")

    figure6_plot_axes.set_ylabel("integral")

    figure6_plot_axes.set_title(
        f"Smooth random walk at lambda={figure6_lambda_ui.value:g}"
    )

    figure6_plot_image = figure_to_rgba_array(figure6_plot_figure)

    plt.close(figure6_plot_figure)

    mo.vstack(
        [
            mo.md(
                rf"""
                ## Figure 6 Reproduction

                The paper states that the same seed is reused across $\lambda = 1/5$, $1/25$, and $1/125$ so the panels are nested refinements of one realization. This notebook uses a Chebfun-like low-to-high coefficient ordering with seed `{figure6_seed}` to stay closer to the staged OCR panels.
                """
            ),
            figure6_lambda_ui,
            mo.hstack(
                [
                    mo.image(
                        figure6_selected_truth,
                        width="49%",
                        height=420,
                        caption="Paper panel",
                        style={"object-fit": "contain"},
                    ),
                    mo.image(
                        figure6_plot_image,
                        width="49%",
                        height=420,
                        caption="Notebook reproduction",
                        style={"object-fit": "contain"},
                    ),
                ],
                justify="space-between",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assignment 3: Geometric Random Walk

    The two stochastic models are

    \[
    dX = X\,dW
    \qquad \text{(Itô)}
    \]

    and

    \[
    dX = X \circ dW
    \qquad \text{(Stratonovich)}.
    \]

    The Stratonovich equation is equivalent to the Itô SDE

    \[
    dX = \frac{1}{2}X\,dt + X\,dW.
    \]

    With $X(0)=X_0$, the exact solutions are

    \[
    X_{\mathrm{Ito}}(t) = X_0 \exp\left(W_t - \frac{t}{2}\right),
    \qquad
    X_{\mathrm{Strat}}(t) = X_0 \exp(W_t).
    \]

    Therefore the required moments are

    \[
    \mathbb{E}[X_{\mathrm{Ito}}(t)] = X_0,
    \qquad
    \mathbb{E}[X_{\mathrm{Ito}}(t)^2] = X_0^2 e^t,
    \]

    and

    \[
    \mathbb{E}[X_{\mathrm{Strat}}(t)] = X_0 e^{t/2},
    \qquad
    \mathbb{E}[X_{\mathrm{Strat}}(t)^2] = X_0^2 e^{2t}.
    \]

    For the smooth random ODE

    \[
    \frac{dX_\lambda}{dt} = X_\lambda f_\lambda(t),
    \]

    we have

    \[
    X_\lambda(t) = X_0 \exp\left(\int_0^t f_\lambda(s)\,ds\right).
    \]

    Because the integral of big smooth random forcing converges to Brownian motion, the limiting behavior should match the **Stratonovich** theory.
    """)
    return


@app.cell
def _(np, simulate_sde_ensembles, simulate_smooth_ode_ensembles):
    stochastic_time_grid, ito_path_ensemble, strat_path_ensemble = (
        simulate_sde_ensembles(
            num_paths=4000,
            t_end=1.5,
            dt=1e-3,
            x0=1.0,
            seed=2026,
        )
    )

    smooth_lambda_values = [0.25, 0.1, 0.05, 0.025]

    smooth_time_grid, smooth_path_ensemble_dict = simulate_smooth_ode_ensembles(
        smooth_lambda_values,
        num_paths=1200,
        t_end=1.5,
        num_points=1501,
        x0=1.0,
        seed=11,
    )

    ito_theory_curves = {
        "mean": np.ones_like(stochastic_time_grid),
        "second": np.exp(stochastic_time_grid),
    }

    strat_theory_curves = {
        "mean": np.exp(0.5 * stochastic_time_grid),
        "second": np.exp(2.0 * stochastic_time_grid),
    }

    smooth_moment_curves = {}

    ito_empirical_curves = {
        "mean": ito_path_ensemble.mean(axis=0),
        "second": (ito_path_ensemble**2).mean(axis=0),
    }

    strat_empirical_curves = {
        "mean": strat_path_ensemble.mean(axis=0),
        "second": (strat_path_ensemble**2).mean(axis=0),
    }

    for smooth_lambda_value in smooth_lambda_values:
        smooth_lambda_paths = smooth_path_ensemble_dict[smooth_lambda_value]
        smooth_moment_curves[smooth_lambda_value] = {
            "mean": smooth_lambda_paths.mean(axis=0),
            "second": (smooth_lambda_paths**2).mean(axis=0),
        }
    return (
        ito_empirical_curves,
        ito_path_ensemble,
        ito_theory_curves,
        smooth_lambda_values,
        smooth_moment_curves,
        smooth_path_ensemble_dict,
        stochastic_time_grid,
        strat_empirical_curves,
        strat_path_ensemble,
        strat_theory_curves,
    )


@app.cell(hide_code=True)
def _(
    ito_empirical_curves,
    ito_theory_curves,
    mo,
    moment_lambda_ui,
    moment_metric_ui,
    plt,
    show_empirical_ui,
    smooth_moment_curves,
    stochastic_time_grid,
    strat_empirical_curves,
    strat_theory_curves,
):
    moment_plot_figure, moment_plot_axes = plt.subplots(figsize=(6.6, 4.1))

    chosen_metric_key = moment_metric_ui.value

    chosen_lambda_value = moment_lambda_ui.value

    chosen_metric_label = (
        r"$\mathbb{E}[X(t)]$"
        if chosen_metric_key == "mean"
        else r"$\mathbb{E}[X(t)^2]$"
    )

    moment_plot_axes.plot(
        stochastic_time_grid,
        ito_theory_curves[chosen_metric_key],
        color="#9c6644",
        lw=2,
        ls="--",
        label="Itô theory",
    )

    moment_plot_axes.plot(
        stochastic_time_grid,
        strat_theory_curves[chosen_metric_key],
        color="#1d3557",
        lw=2,
        ls="--",
        label="Stratonovich theory",
    )

    moment_plot_axes.plot(
        stochastic_time_grid,
        smooth_moment_curves[chosen_lambda_value][chosen_metric_key],
        color="#1b4332",
        lw=2.2,
        label=rf"smooth random ODE, $\lambda={chosen_lambda_value}$",
    )

    if show_empirical_ui.value:
        moment_plot_axes.plot(
            stochastic_time_grid,
            ito_empirical_curves[chosen_metric_key],
            color="#bc6c25",
            lw=1.3,
            alpha=0.9,
            label="Itô simulation",
        )
        moment_plot_axes.plot(
            stochastic_time_grid,
            strat_empirical_curves[chosen_metric_key],
            color="#457b9d",
            lw=1.3,
            alpha=0.9,
            label="Stratonovich simulation",
        )

    moment_plot_axes.set_xlabel("t")

    moment_plot_axes.set_ylabel(chosen_metric_label)

    moment_plot_axes.set_title("Moment comparison for geometric random walk")

    if chosen_metric_key == "second":
        moment_plot_axes.set_yscale("log")

    moment_plot_axes.legend()

    mo.vstack(
        [
            mo.md(
                f"""
                ## Ensemble-Moment Comparison

                The controls choose which smooth-ODE curve to compare against the analytical Itô and Stratonovich predictions.

                {"For the **mean**, the Itô theory is flat because $\\mathbb{E}[X(t)] = X_0 = 1$ for $dX = X\\,dW$ with $X_0=1$." if chosen_metric_key == "mean" else "For the **second moment**, the Itô prediction is not flat; it grows like $e^t$ while the Stratonovich prediction grows like $e^{2t}$."}
                """
            ),
            mo.hstack(
                [moment_metric_ui, moment_lambda_ui, show_empirical_ui],
                justify="start",
            ),
            mo.mpl.interactive(moment_plot_figure),
        ]
    )
    return


@app.cell
def _(mo, smooth_lambda_values):
    snapshot_lambda_ui = mo.ui.slider(
        steps=smooth_lambda_values,
        value=smooth_lambda_values[-1],
        show_value=True,
        label="Smooth-ODE lambda in histogram",
    )

    snapshot_region_ui = mo.ui.range_slider(
        start=0.0,
        stop=6.0,
        step=0.1,
        value=[0.8, 1.2],
        show_value=True,
        label="Highlight x-region",
    )
    return snapshot_lambda_ui, snapshot_region_ui


@app.cell(hide_code=True)
def _(
    ito_path_ensemble,
    mo,
    np,
    plt,
    smooth_path_ensemble_dict,
    snapshot_lambda_ui,
    snapshot_region_ui,
    snapshot_time_ui,
    stochastic_time_grid,
    strat_path_ensemble,
):
    snapshot_index = int(
        round(
            snapshot_time_ui.value
            / (stochastic_time_grid[1] - stochastic_time_grid[0])
        )
    )

    snapshot_hist_bins = np.linspace(
        0.0, np.percentile(strat_path_ensemble[:, snapshot_index], 99.5), 70
    )

    snapshot_plot_figure, snapshot_plot_axes = plt.subplots(figsize=(6.6, 4.1))

    snapshot_plot_axes.hist(
        ito_path_ensemble[:, snapshot_index],
        bins=snapshot_hist_bins,
        density=True,
        alpha=0.35,
        color="#bc6c25",
        label="Itô ensemble",
    )

    snapshot_plot_axes.hist(
        strat_path_ensemble[:, snapshot_index],
        bins=snapshot_hist_bins,
        density=True,
        alpha=0.35,
        color="#457b9d",
        label="Stratonovich ensemble",
    )

    snapshot_plot_axes.hist(
        smooth_path_ensemble_dict[snapshot_lambda_ui.value][:, snapshot_index],
        bins=snapshot_hist_bins,
        density=True,
        histtype="step",
        lw=2,
        color="#1b4332",
        label=rf"smooth ODE, $\lambda={snapshot_lambda_ui.value}$",
    )

    snapshot_plot_axes.axvspan(
        snapshot_region_ui.value[0],
        snapshot_region_ui.value[1],
        color="#264653",
        alpha=0.12,
        label="selected region",
    )

    snapshot_plot_axes.set_xlabel("X")

    snapshot_plot_axes.set_ylabel("density")

    snapshot_plot_axes.set_title(
        f"Distribution snapshot at t={snapshot_time_ui.value}"
    )

    snapshot_plot_axes.legend()

    mo.vstack(
        [
            mo.md(
                r"""
                ## Distribution Snapshots

                The histogram view makes the qualitative separation visible: the smooth random ODE clusters with the Stratonovich dynamics rather than the Itô dynamics.
                """
            ),
            mo.hstack(
                [snapshot_time_ui, snapshot_lambda_ui, snapshot_region_ui],
                justify="start",
            ),
            mo.mpl.interactive(snapshot_plot_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    ito_path_ensemble,
    mo,
    smooth_path_ensemble_dict,
    snapshot_lambda_ui,
    snapshot_region_ui,
    snapshot_time_ui,
    stochastic_time_grid,
    strat_path_ensemble,
):
    snapshot_region_index = int(
        round(
            snapshot_time_ui.value
            / (stochastic_time_grid[1] - stochastic_time_grid[0])
        )
    )

    region_left, region_right = snapshot_region_ui.value

    region_models = {
        "Ito": ito_path_ensemble[:, snapshot_region_index],
        "Stratonovich": strat_path_ensemble[:, snapshot_region_index],
        f"Smooth lambda={snapshot_lambda_ui.value}": smooth_path_ensemble_dict[
            snapshot_lambda_ui.value
        ][:, snapshot_region_index],
    }

    region_rows = []

    for region_name, region_values in region_models.items():
        region_mask = (region_values >= region_left) & (
            region_values <= region_right
        )
        inside_values = region_values[region_mask]
        region_rows.append(
            {
                "model": region_name,
                "count_in_region": int(region_mask.sum()),
                "probability_mass": round(float(region_mask.mean()), 4),
                "conditional_mean": round(
                    float(inside_values.mean())
                    if inside_values.size
                    else float("nan"),
                    4,
                ),
                "conditional_std": round(
                    float(inside_values.std())
                    if inside_values.size
                    else float("nan"),
                    4,
                ),
            }
        )

    mo.vstack(
        [
            mo.md(
                f"""
        ### Selected-Region Summary

        The table below reports how much probability mass falls inside the highlighted region $[{region_left:.2f}, {region_right:.2f}]$ at time $t={snapshot_time_ui.value}$.
        """
            ),
            mo.ui.table(
                region_rows,
                selection=None,
                pagination=False,
                label="Region summary",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    - The Fourier-series construction reproduces the structure of Figure 1 without Chebfun.
    - The big normalization produces integrated paths consistent with Brownian scaling and the Figure 6 random-walk picture.
    - In the geometric random walk, the smooth random ODE aligns with the **Stratonovich** benchmark as $\lambda$ decreases.

    All figures in this notebook are generated from local code and local project documents. The main diagnostics are exposed through marimo UI elements so the notebook can be explored interactively in `marimo edit`, presented as slides, or exported as HTML.
    """)
    return


if __name__ == "__main__":
    app.run()
