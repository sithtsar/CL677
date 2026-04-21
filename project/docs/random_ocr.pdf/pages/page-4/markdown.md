SMOOTH RANDOM FUNCTIONS

real periodic smooth random function is the real part of a complex one. Equivalently, it is a function

$$
f(x) = a_0 + \sqrt{2} \sum_{j=1}^{m} \left[ a_j \cos \left(\frac{2\pi jx}{L}\right) + b_j \sin \left(\frac{2\pi jx}{L}\right) \right], \quad m = \lfloor L / \lambda \rfloor, \tag{2.2}
$$

where each $a_j$ and each $b_j$ is an independent sample from $N(0,1/(2m+1))$.

As usual, $N(\mu, V)$ denotes the real normal distribution of mean $\mu$ and variance $V$, and $\lfloor \cdot \rfloor$ is the floor function. According to standard terminology, $f$ is a trigonometric polynomial of degree $m$ [52]. To verify that (2.2) is equivalent to the real part of (2.1), we can write $c_j = \alpha_j + i\beta_j$, where $\alpha_j$ and $\beta_j$ are independent samples from $N(0,1/(2m+1))$, and note that $\exp(2\pi ijx/L) = \cos(2\pi jx/L) + i\sin(2\pi jx/L)$. Grouping together real terms, we find that the real part of (2.1) can be expanded as

$$
\alpha_0 + \sum_{j=1}^{m} \left[ (\alpha_j + \alpha_{-j}) \cos \left(\frac{2\pi jx}{L}\right) + (-\beta_j + \beta_{-j}) \sin \left(\frac{2\pi jx}{L}\right) \right].
$$

Since $\alpha_j$ and $\alpha_{-j}$ are independent samples from $N(0,1/(2m+1))$ for $j \geq 1$, their sum is a sample from $N(0,2/(2m+1))$, hence equivalent to $\sqrt{2}a_j$ with $a_j$ from $N(0,1/(2m+1))$; similarly for the terms involving $\beta_j$ and $\beta_{-j}$.

A theorem summarizes some of the properties of these functions. We say that a periodic function is $k$-band-limited if it can be written as a Fourier series with wave numbers confined to $[-k, k]$.

THEOREM 2.2. A periodic smooth random function $f$ (whether real or complex) is $L$-periodic, entire, and $(2\pi/\lambda)$-band-limited. The stochastic process from which $f$ is a sample is stationary (i.e., it has a distribution that is translation-invariant), with values $f(x)$ at each $x$ distributed according to $N(0,1) + iN(0,1)$ in the complex case and $N(0,1)$ in the real case.

Proof. The periodicity is immediate from (2.1) or (2.2), and $f$ is entire (i.e., analytic throughout the complex $x$-plane) since it is a finite sum of complex exponentials or sines and cosines. Since $|j| \leq m \leq L/\lambda$, the maximum value of the coefficients $|2\pi j/L|$ in (2.1) or (2.2) is bounded by $2\pi/\lambda$, so $f$ is $(2\pi/\lambda)$-band-limited. Stationarity of the stochastic process follows from (2.1) since translating a function $c_j \exp(2\pi ijx/L)$ amounts to changing the argument but not the modulus of $c_j$ and the distribution $N(0,1/(2m+1)) + iN(0,1/(2m+1))$ is argument-invariant. Hence any translated process has an identical distribution to the original. The same argument-invariance of $N(0,1/(2m+1)) + iN(0,1/(2m+1))$ also ensures that the sum in (2.1) is distributed according to $N(0,1) + iN(0,1)$ at each point $x$, and its real part accordingly has the distribution $N(0,1)$.

These definitions of random functions are rooted in the Fourier domain: they describe a random function as a sum of Fourier modes with random amplitudes. (We may think either of random complex amplitudes or of random real amplitudes coupled with random phases.) Equivalently, as suggested in Figure 2, we can construct random functions in the spatial domain. The essential point here is that there is an equivalence between the $2m + 1$ Fourier series coefficients $\{c_j\}$ of (2.1) and the $2m + 1$ function values

$$
d_j = f(x_j), \quad -m \leq j \leq m,
$$