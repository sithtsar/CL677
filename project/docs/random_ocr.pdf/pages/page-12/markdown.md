measure associates exit probabilities of Brownian paths with solutions of the Laplace equation. The basis of such connections is the fact that the density of an ensemble of Brownian motions obeys a diffusion PDE, with spreading at a rate characterized by $\sqrt{t}$. This observation goes all the way back to Einstein.

The difficulty with the pointwise approach to stochasticity, however, is that it is highly technical. In continuum mechanics we can write down the gas laws or the Navier–Stokes equations without discussing the underlying molecules, but it is not possible to state the principles of stochastic analysis so easily. Stochastic analysis requires special foundations, and they are technically advanced. These in turn require special notations and special numerical methods, which are different from the familiar methods of nonstochastic numerical computation *[24]*. This becomes an issue particularly in the context of differential equations, the subject of the next section.

Ultimately, anyone working in this field will need to deal with the technicalities. Smooth random functions, however, provide an elementary way to get started. Since $\lambda>0$ always, they build just on ordinary calculus and ordinary numerical methods (quadrature in this section, solution of ODEs in the next).

The possibility of defining Brownian paths via Fourier series with random coefficients goes back a long way. As discussed by Kahane *[21, 22]*, Wiener considered such series in *[48]*, and the discussion was generalized by him and Paley and Zygmund in several papers including *[36]*. As these authors noted, the integral of the series (2.1) or (2.2) contains coefficients mollified by the factor $1/j$. Thus, for example, integration of (2.1) gives

\[ \int_{0}^{x}f(s)ds=c_{0}x+\frac{L}{2\pi i}\sum_{\begin{subarray}{c}j=-m\\
j\neq 0\end{subarray}}^{m}\frac{c_{j}}{j}\left[\exp\left(\frac{2\pi ijx}{L}\right)-1\right]. \]

With $m=\infty$ this becomes an infinite series whose convergence is not guaranteed, since $O(j^{-1})$ coefficients do not decrease quite fast enough, but if the coefficients are random, that ensures convergence with probability 1.

The central property of big smooth random functions is that their integrals converge to standard Brownian paths as $\lambda\to 0$. (The term standard refers to a normalization. The variance of the distribution of a Brownian path $W(t)$ is equal to $Ct$ for some constant $C$, and the standard choices are $C=1$ for the real case and $C=2$ for the complex case.) We designate this as a theorem for clarity, but the statement below is not really precise, and indeed this paper does not present any of the definitions and details needed for rigorous stochastic analysis. For a full treatment, we recommend Chapter 16 of Kahane’s book *[21]*, particularly Theorem 2 on p. 236. Ultimately this result is due to Wiener.

###### Theorem 4.3.

As $\lambda\to 0$, indefinite integrals of big smooth random functions (whether real or complex, periodic or nonperiodic) converge with probability $1$ to standard Brownian paths.

For a fascinating presentation of the mathematical properties of Brownian motion, see the book *[32]* mentioned earlier, and a more advanced treatment can be found in *[39]*. The physical side of the subject is presented in *[15]*, also with a discussion of applications in finance.

## 5 Smooth Random ODEs

Having defined smooth random functions, we can use them as forcing functions, or as coefficients, in ordinary differential equations (ODEs). Sometimes this is interesting for fixed $\lambda$, typically with the standard nor