# CurveFitModels.jl Documentation

A Julia package containing model functions for curve fitting with [CurveFit.jl](https://github.com/SciML/CurveFit.jl).

All model functions follow the CurveFit.jl convention: `fn(parameters, x)` where parameters come first.

## Example Usage

Fit a Lorentzian to a molecular absorption peak:

```julia
using CurveFit
using CurveFitModels

# Simulated IR absorption peak near 2100 cm⁻¹ (e.g., C≡O stretch)
ν = range(2050, 2150, length=100)  # wavenumber (cm⁻¹)
true_params = [5.0, 2100.0, 12.0]  # [amplitude, center, FWHM]
y = lorentzian_fwhm(true_params, ν) .+ 0.003 .* randn(length(ν))

# Fit with initial guess
p0 = [4.7, 2098.0, 15.0]  # initial guess for [A, ν₀, Γ]
prob = NonlinearCurveFitProblem(lorentzian_fwhm, p0, ν, y)
sol = solve(prob)

# Extract fitted parameters
A, ν₀, Γ = sol.u
```

![Lorentzian fit example](assets/lorentzian_fit_example.png)
