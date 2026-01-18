# CurveFitModels.jl Documentation

A Julia package containing model functions for curve fitting with [CurveFit.jl](https://github.com/SciML/CurveFit.jl).

All model functions follow the CurveFit.jl convention: `fn(parameters, x)` where parameters come first.

## Example Usage

```julia
using CurveFit
using CurveFitModels

# Generate some noisy data
t = collect(0.0:0.1:5.0)
p_true = [2.0, 0.5, 0.1]  # [A, τ, y₀]
y_data = single_exponential(p_true, t) .+ 0.05 * randn(length(t))

# Fit the data
p0 = [1.5, 0.3, 0.0]  # initial guess
prob = NonlinearCurveFitProblem(single_exponential, p0, t, y_data)
sol = solve(prob)
p_fit = coef(sol)
```
