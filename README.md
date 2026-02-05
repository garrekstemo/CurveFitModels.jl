# CurveFitModels.jl

[![CI](https://github.com/garrekstemo/CurveFitModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/garrekstemo/CurveFitModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/garrekstemo/CurveFitModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/garrekstemo/CurveFitModels.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://garrekstemo.github.io/CurveFitModels.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://garrekstemo.github.io/CurveFitModels.jl/dev)

A Julia package providing model functions for curve fitting with [CurveFit.jl](https://github.com/SciML/CurveFit.jl). Originally developed with spectroscopy applications in mind, but useful for any nonlinear curve fitting task.

All model functions follow the CurveFit.jl convention: `fn(parameters, x)` where parameters come first.

## Available Models

**Lineshapes**: `gaussian`, `lorentzian`, `pseudo_voigt`, `power_law`, `logistic`

**Temporal**: `single_exponential`, `stretched_exponential`, `n_exponentials`, `sine`, `damped_sine`

**Oscillator models**: `lorentz_oscillator`, `dielectric_real`, `dielectric_imag`

**Composition**: `poly`, `combine`, `gaussian2d`

## Installation

```julia
using Pkg
Pkg.add("CurveFitModels")
```

## Example

Fit a Lorentzian to a molecular absorption peak:

```julia
using CurveFit
using CurveFitModels
using CairoMakie

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
y_fit = lorentzian(sol.u, ν)

# Plot
fig = Figure(size=(500, 400))
ax = Axis(fig[1, 1], xlabel="Wavenumber (cm⁻¹)", ylabel="Absorbance")
scatter!(ax, ν, y, color=:black, markersize=5, label="Data")
lines!(ax, ν, y_fit, color=:crimson, linewidth=2, label="Fit")
axislegend(ax, position=:rt)

# Annotations
text!(ax, 2055, 0.035, text="ν₀ = $(round(ν₀, digits=1)) cm⁻¹\nΓ = $(round(Γ, digits=1)) cm⁻¹\nA = $(round(A, digits=2))", fontsize=12)

fig
```

![Lorentzian fit example](docs/src/assets/lorentzian_fit_example.png)

## Model Composition

Combine models with polynomial baselines for simultaneous fitting:

```julia
# poly(p, x) evaluates c₀ + c₁x + c₂x² + ...
model = combine(lorentzian, 3, poly, 2)  # lorentzian + linear baseline

p0 = [A, x0, Γ, c0, c1]  # peak params + baseline params
prob = NonlinearCurveFitProblem(model, p0, x, y)
sol = solve(prob)
```

## Helper Functions

```julia
# Width conversion
fwhm = sigma_to_fwhm(σ)   # Gaussian σ → FWHM
σ = fwhm_to_sigma(fwhm)   # FWHM → Gaussian σ

# Area calculation
area = gaussian_area(A, σ)    # A × σ × √(2π)
area = lorentzian_area(A, Γ)  # A × π × Γ / 2
```

See the [documentation](https://garrekstemo.github.io/CurveFitModels.jl/stable) for full details on all functions.
