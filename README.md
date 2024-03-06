# stochasticclock

![](https://raw.githubusercontent.com/Mitrxs/Stochastic-clock/main/example_figures/Example_illustration.png)

A module calculating the stochastic deviations in timepoints for atomic clocks. 

The code is an application of the theory presented in Galleani et al. (2003), doi:10.1088/0026-1394/40/3/305.

The module's current functionality calculates stochastic deviations using the exact iterative solution to the stochastic differential equation in `Galleani_exact()`

$$\begin{equation}
    \mathbf{X}(t_{n+1}) =
        \begin{pmatrix}
            1 & \delta t \\\ 
            0 & 1 
        \end{pmatrix}
    \mathbf{X}(t_n) +
        \begin{pmatrix}
            \delta t \mu_1 + \frac{1}{2} \delta t^2 \mu_2 \\\ 
            \delta t \mu_2
        \end{pmatrix} 
    + \mathbf{J}(t_n)
\end{equation}$$ 

$$\begin{equation}
    \mathbf{J}(t_n) \sim \mathcal{N} \bigg( \mathbf{0},
        \begin{bmatrix}
            \sigma_1^2 \delta t + \frac{1}{3} \sigma_2^2 \delta t^3 & \frac{1}{2}\sigma_2^2 \delta t^2 \\\ 
            \frac{1}{2}\sigma_2^2 \delta t^2 & \sigma_2^2 \delta t 
        \end{bmatrix} 
    \bigg)
\end{equation}$$ 

Stochastic deviations can be visualised using `clock_error()`, and their distributions simulated with `deviation_distribution()`.

Please consult the Jupyter notebook for examples.

