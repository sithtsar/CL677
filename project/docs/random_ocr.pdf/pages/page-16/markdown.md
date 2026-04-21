is only additive noise, they are identical. Details can be found in many texts on stochastic analysis, such as *[15, 24, 35]*. In a word, smooth random ODEs as we have set them up correspond to the Stratonovich formulation. Equation (5.1), for example, is a smooth approximation to this SDE written in the standard notation:

$dX_{t}=(X_{t}-X_{t}^{3})dt+0.5dW_{t}.$

In this case, as there is only additive noise, the Itô and Stratonovich settings coincide. An example with multiplicative noise is (5.4), and in this case the convention is to include a “$\circ$” symbol to indicate that the formulation is Stratonovich. Equation (5.4) is a smooth approximation to this SDE:

$dX_{t}=(X_{t}-X_{t}^{2})dt+\sigma X_{t}\circ dW_{t}.$

The central property of solutions of ODEs containing a big smooth random function is that they converge to solutions of Stratonovich SDEs as $\lambda\to 0$. As with Theorem 4.3, we state this as a theorem, but the statement is not precise, for that would require details of stochastic analysis. The result is essentially due to Wong and Zakai in a pair of papers from the mid-1960s *[50, 51]*, of which an account is given in the book by Wong and Hajek *[49]*. “Wong–Zakai theory” is more general than this result, however, not requiring smoothness in the random functions, a property that would be regarded as needlessly restrictive by many mathematicians. Our recommended reference on this material is the paper *[43]* by Sussmann and its short summary *[42]*, which connect SDEs and ODEs at the level of individual solutions paths (see also *[25]*). A popular reference for SDEs is *[35]*, in which these matters are briefly discussed. For recent developments, see *[18]* and *[54]*. A fundamental generalization of some of these ideas is the theory of rough paths introduced by Lyons *[14, 28]*.

###### Theorem 5.1.

As $\lambda\to 0$, solutions to random ODEs containing a big smooth random function converge with probability $1$ to solutions of Stratonovich SDEs.

Note that the theorem allows for just one big smooth random function, not several. When more than one random variable is involved, the relationship with the theory of SDEs is not as simple. New issues also arise when there is more than one independent variable, i.e., with stochastic PDEs. For information on these and many other matters in the approximation of stochastic systems by differential equations, see *[9, 10, 23, 37, 43, 54]*.

## 6. Smooth Random Functions and Gaussian Processes

A smooth random function is a sample path from a particular Gaussian process. Informally, a Gaussian process is a stochastic process depending on a continuous variable $t$ in which for each fixed $t$, the value is a Gaussian random variable, and moreover, for any finite set $t_{1},\ldots,t_{k}$, the joint distribution is multivariate Gaussian. Gaussian processes are an old idea, but in recent decades interest in them has increased greatly with the advance of Bayesian reasoning in general and machine learning in particular *[19, 29, 34, 38, 41]*.

A Gaussian process is determined by a mean function $\mu(t)$ (here, just the zero function) and a covariance function $C(t,t^{\prime})$, defined (when the mean is zero) as the expected value of $\overline{f(t^{\prime})}f(t)$. If $C(t,t^{\prime})$ depends just on $t^{\prime}-t$, the process is stationary (another term is homogeneous). For smooth random functions, we showed in section 2 that $C(t,t^{\prime})$ is the Dirichlet kernel (2.4): $C(t,t^{\prime})=D(t^{\prime}-t)$, or $2D(t^{\prime}-t)$ in the complex case. Other choices of covariance function also make sense and, indeed, they have advantages. A particularly attractive choice is a Gaussian kernel,