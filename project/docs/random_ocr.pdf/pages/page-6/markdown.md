data point, also known as a periodic sinc function or the Dirichlet kernel. From here we see that any $(2\pi/\lambda)$-band-limited $L$-periodic function can be specified in the form

$f(x)=\sum_{j=-m}^{m}d_{{}_{j}}D(x-jh),\quad m=\lfloor L/\lambda\rfloor.$

The observations above depend only on the fact that $F$ is nonsingular. The equivalence we need for smooth random functions follows from the further fact that $F$ is a multiple of a unitary matrix. In the definition (2.1), the variables $c_{{}_{j}}$ are independent samples from $N(0,1/(2m+1))+iN(0,1/(2m+1))$. By a standard result of multivariate statistics, this is the same as saying that their joint probability distribution is

$p(\mathbf{c})=C\exp(-(2m+1)\|\mathbf{c}\|^{2}/2),$

where $\|\cdot\|$ is the 2-norm and the constant $C>0$ normalizes the total probability to 1. Since $\mathbf{d}=F\mathbf{c}$ and $F$ is $\sqrt{2m+1}$ times a unitary matrix, this is equivalent to saying that the joint probability distribution of the values $\{d_{{}_{j}}\}$ is

$p(\mathbf{d})=C^{\prime}\exp(-\|\mathbf{d}\|^{2}/2),$

where $C^{\prime}$ is again a normalization constant. Therefore the values $d_{{}_{j}}$ are independent samples from $N(0,1)+iN(0,1)$. This observation establishes that the following definition is equivalent to the earlier one.

###### Definition 2.3.

A *real or complex periodic smooth random function* for given $\lambda,L>0$ is a function

$f(x)=\sum_{j=-m}^{m}d_{{}_{j}}D(x-jh),\quad m=\lfloor L/\lambda\rfloor,$ (2.5)

with $h$ defined by (2.3), where each $d_{{}_{j}}$ is an independent sample from $N(0,1)$ or $N(0,1)+iN(0,1)$, respectively.

It is worth emphasizing this equivalence. Fourier series with random coefficients are the same as trigonometric interpolants through random data values. Though we have made use of the particular choice (2.3) of gridpoints $x_{{}_{j}}$, it follows from translation-invariance that this choice does not matter. Translation to any other equispaced grid will produce the same distribution of smooth random functions.

Periodic smooth random functions define a Gaussian process with mean zero, as we shall discuss in section 6. As a taste of this interpretation, we note here that translation-invariance reveals how values of a smooth random function depend on one another between gridpoints. The covariance of the stochastic process is the function $C(x,y)$ defined as the expected value of the product $f(x)f(y)$, or $\overline{f(x)}f(y)$ in the complex case. Because of stationarity, this is equal to the expected value of $\overline{f(x-y)}f(0)$. By (2.5), $f(0)$ reduces to the random number $d_{{}_{0}}$, and since the coefficients $d_{{}_{j}}$ are uncorrelated with $d_{0}$ for $j\neq 0$, it follows from (2.5) that the expected value of $\overline{f(x-y)}f(0)$ reduces to the expected value of $\overline{d_{{}_{0}}}d_{{}_{0}}D(x-y)$, that is, $D(x-y)$ in the real case and $2D(x-y)$ in the complex case.

Figure 3 shows an example of one kind of use of smooth random functions. In many applications one would like to take random initial data to explore the typical