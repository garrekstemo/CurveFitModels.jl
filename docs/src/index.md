# CurveFitModels.jl Documentation

A Julia package containing model functions for curve fitting with [CurveFit.jl](https://github.com/SciML/CurveFit.jl).

All model functions follow the CurveFit.jl convention: `fn(parameters, x)` where parameters come first.

## Parameters

Both `gaussian` and `lorentzian` use consistent parameterization:

| Parameter | Gaussian | Lorentzian |
|-----------|----------|------------|
| `A` | Peak amplitude | Peak amplitude |
| `x₀` | Center position | Center position |
| Width | `σ` (std dev) | `Γ` (FWHM) |
| `y₀` | Offset (optional) | Offset (optional) |

Helper functions:

```julia
fwhm = sigma_to_fwhm(σ)       # Gaussian σ → FWHM
σ = fwhm_to_sigma(fwhm)       # FWHM → Gaussian σ
area = gaussian_area(A, σ)    # A × σ × √(2π)
area = lorentzian_area(A, Γ)  # A × π × Γ / 2
```

## Model Composition

Combine models with polynomial baselines for simultaneous fitting:

```julia
# poly(p, x) evaluates c₀ + c₁x + c₂x² + ...
model = combine(lorentzian, 3, poly, 2)  # lorentzian + linear baseline

p0 = [A, x0, Γ, c0, c1]  # peak params + baseline params
prob = NonlinearCurveFitProblem(model, p0, x, y)
sol = solve(prob)
```

## Example Usage

Fit a Lorentzian to a molecular absorption peak:

```julia
using CurveFit
using CurveFitModels

# Simulated IR absorption peak near 2100 cm⁻¹ (e.g., C≡O stretch)
ν = range(2050, 2150, length=100)  # wavenumber (cm⁻¹)
true_params = [5.0, 2100.0, 12.0]  # [amplitude, center, FWHM]
y = lorentzian(true_params, ν) .+ 0.003 .* randn(length(ν))

# Fit with initial guess
p0 = [4.7, 2098.0, 15.0]  # initial guess for [A, ν₀, Γ]
prob = NonlinearCurveFitProblem(lorentzian, p0, ν, y)
sol = solve(prob)

# Extract fitted parameters
A, ν₀, Γ = sol.u
```

![Lorentzian fit example](assets/lorentzian_fit_example.png)
