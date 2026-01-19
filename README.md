# CurveFitModels.jl

A Julia package containing model functions for curve fitting with [CurveFit.jl](https://github.com/SciML/CurveFit.jl).

All model functions follow the CurveFit.jl convention: `fn(parameters, x)` where parameters come first.

## Installation

Since this package is not registered, install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/garrekds/CurveFitModels.jl")
```

## Example

Fit a Gaussian to noisy data:

```julia
using CurveFit
using CurveFitModels

# Generate noisy data
x = range(-5, 5, length=100)
true_params = [1.0, 0.0, 1.0]  # amplitude, center, σ
y = gaussian(true_params, x) .+ 0.05 .* randn(length(x))

# Fit with initial guess
p0 = [0.8, 0.1, 1.2]  # initial guess for [A, x₀, σ]
prob = NonlinearCurveFitProblem(gaussian, p0, x, y)
sol = solve(prob)

println("Fitted parameters: ", sol.u)
println("Evaluate fit at x=0: ", sol(0.0))

See the [CurveFit.jl documentation](https://github.com/SciML/CurveFit.jl) for more details on fitting.
```

