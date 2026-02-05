# CurveFitModels.jl

A Julia package providing model functions for curve fitting with [CurveFit.jl](https://github.com/SciML/CurveFit.jl). Originally developed with spectroscopy applications in mind, but useful for any nonlinear curve fitting task.

All model functions follow the CurveFit.jl convention: `fn(parameters, x)` where parameters come first.

## Available Models

**Lineshapes**: [`gaussian`](@ref), [`lorentzian`](@ref), [`pseudo_voigt`](@ref), [`power_law`](@ref), [`logistic`](@ref), [`gaussian2d`](@ref)

**Temporal**: [`single_exponential`](@ref), [`stretched_exponential`](@ref), [`n_exponentials`](@ref), [`sine`](@ref), [`damped_sine`](@ref)

**Oscillator models**: [`lorentz_oscillator`](@ref), [`dielectric_real`](@ref), [`dielectric_imag`](@ref)

**Utilities**: [`poly`](@ref), [`combine`](@ref)

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
