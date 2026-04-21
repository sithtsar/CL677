# Project A Implementation Spec

## Purpose

This document turns the course brief in `docs/project_A.md`, Assignment 3 in
`docs/CL677_A3.pdf`, and the paper `docs/random.pdf` into an implementation
checklist for the hybrid marimo notebook deliverable in `notebook.py`.

## Required outcomes

- Build one marimo notebook that works as both:
  - a reproducible analysis notebook in `marimo edit` / `marimo run`
  - a presentation-friendly deck via `marimo export pdf --as=slides`
  - a portable browser artifact via `marimo export html-wasm`
- Reproduce paper Figure 1 and Figure 6 without using Chebfun helpers.
- Show numerically that smooth random forcing in the geometric random walk tends
  to the Stratonovich interpretation as the correlation scale `lambda -> 0`.

## Mathematical requirements

### Smooth random functions

- Implement a real-valued periodic smooth random function on an interval of
  length `L` using the truncated Fourier series

  `f(x) = a0 + sqrt(2) * sum_{j=1}^m [a_j cos(2 pi j x / L) + b_j sin(2 pi j x / L)]`

  with `m = floor(L / lambda)`.
- Generate the coefficients from scratch with independent Gaussian draws.
- Support both normalizations from the paper:
  - standard normalization for function-value plots:
    `a_j, b_j ~ N(0, 1 / (2m + 1))`
  - big normalization for Brownian/SDE forcing:
    `a_j, b_j ~ N(0, 2 / ((2m + 1) lambda))`
    which is equivalent to multiplying the standard-normalized function by
    `sqrt(2 / lambda)`.

### Brownian approximation

- Implement indefinite integration of a big smooth random function to create a
  smooth random walk.
- Use this to demonstrate Brownian scaling on `[0, 1]`.
- Keep the seed/realization policy nested across decreasing `lambda` values so
  finer approximations extend the same underlying path.

### Geometric random walk theory

- State the two SDEs from Assignment 3:
  - Itô: `dX = X dW`
  - Stratonovich: `dX = X ◦ dW`
- State the Itô-equivalent form of the Stratonovich equation:
  - `dX = (1/2) X dt + X dW`
- Include concise derivations or stated closed forms for the moments:
  - Itô:
    - `E[X(t)] = X0`
    - `E[X(t)^2] = X0^2 exp(t)`
  - Stratonovich:
    - `E[X(t)] = X0 exp(t/2)`
    - `E[X(t)^2] = X0^2 exp(2t)`

## Notebook content checklist

- Title and project brief.
- Problem statement and explicit deliverables.
- Mathematical setup for smooth random functions.
- Implementation details and reusable helper functions.
- Figure 1 reproduction with ground-truth comparison.
- Big normalization and Brownian-convergence diagnostics.
- Figure 6 reproduction with ground-truth comparison.
- Assignment 3 geometric random walk theory and expected moments.
- Numerical experiments comparing:
  - Itô SDE ensemble
  - Stratonovich SDE ensemble
  - smooth random ODE ensemble `dX/dt = X f_lambda(t)`
- Distribution snapshots showing the separation between Itô and Stratonovich.
- Conclusion and reproducibility notes.

## Acceptance criteria

- No Chebfun dependency.
- Only `numpy`, `scipy`, `matplotlib`, and marimo UI/markdown features are
  required.
- The notebook runs top-to-bottom without graph errors.
- Standard-normalized samples have pointwise variance near 1.
- Integrated big-normalized samples show variance growth consistent with
  Brownian motion.
- Figure 1 and Figure 6 match the structure of the paper figures.
- For the geometric random walk, simulated Itô and Stratonovich ensemble
  moments match their analytical formulas.
- As `lambda` decreases, the smooth-random-ODE statistics move toward the
  Stratonovich predictions and away from the Itô-only benchmark.
