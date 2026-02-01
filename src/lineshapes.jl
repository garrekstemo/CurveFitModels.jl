const FWHM_SIGMA_FACTOR = 2 * sqrt(2 * log(2))  # ≈ 2.3548

"""
    sigma_to_fwhm(σ)

Convert Gaussian standard deviation to full width at half maximum.

FWHM = 2√(2ln2) × σ ≈ 2.355 × σ
"""
sigma_to_fwhm(σ) = FWHM_SIGMA_FACTOR * σ

"""
    fwhm_to_sigma(fwhm)

Convert full width at half maximum to Gaussian standard deviation.

σ = FWHM / 2√(2ln2) ≈ FWHM / 2.355
"""
fwhm_to_sigma(fwhm) = fwhm / FWHM_SIGMA_FACTOR

"""
    gaussian_area(A, σ)

Calculate the integrated area under a Gaussian with peak amplitude `A` and standard deviation `σ`.

Area = A × σ × √(2π)
"""
gaussian_area(A, σ) = A * σ * sqrt(2π)

"""
    lorentzian_area(A, Γ)

Calculate the integrated area under a Lorentzian with peak amplitude `A` and FWHM `Γ`.

Area = A × π × Γ / 2
"""
lorentzian_area(A, Γ) = A * π * Γ / 2

"""
    poly(p, x)

Polynomial function for curve fitting. Uses Horner's method via `evalpoly`.

# Arguments
- `p`: Coefficients [c₀, c₁, c₂, ...] for c₀ + c₁x + c₂x² + ...
- `x`: Independent variable

# Example
```julia
poly([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])  # 1 + 2x + 3x²
# returns [1.0, 6.0, 17.0]
```

Combine with other models for simultaneous baseline fitting:
```julia
model(p, x) = lorentzian(p[1:3], x) .+ poly(p[4:5], x)
```
"""
poly(p, x) = evalpoly.(x, Ref(Tuple(p)))

"""
    combine(f1, n1, f2, n2)

Combine two model functions into one by splitting the parameter vector.

# Arguments
- `f1`: First model function with signature `f1(p, x)`
- `n1`: Number of parameters for `f1`
- `f2`: Second model function with signature `f2(p, x)`
- `n2`: Number of parameters for `f2`

Returns a new function `(p, x) -> f1(p[1:n1], x) .+ f2(p[n1+1:n1+n2], x)`.

# Example
```julia
# Lorentzian (3 params) + linear baseline (2 params)
model = combine(lorentzian, 3, poly, 2)

p0 = [1.0, 0.0, 1.0, 0.1, 0.01]  # [A, x0, Γ, c0, c1]
prob = NonlinearCurveFitProblem(model, p0, x, y)
sol = solve(prob)
```
"""
combine(f1, n1, f2, n2) = (p, x) -> f1(p[1:n1], x) .+ f2(p[n1+1:n1+n2], x)

"""
    gaussian(p, x)

Gaussian function with standard deviation parameterization.

# Arguments
- `p`: Parameters [A, x₀, σ] or [A, x₀, σ, y₀]
  - `A`: Amplitude
  - `x₀`: Center position
  - `σ`: Standard deviation
  - `y₀`: Vertical offset (default: 0.0)
- `x`: Independent variable

Use `fwhm_to_sigma` and `sigma_to_fwhm` to convert between σ and FWHM.

```math
\\begin{aligned}
    f(x) = A \\exp\\left(-\\frac{(x - x_0)^2}{2\\sigma^2}\\right) + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Gaussian_function](https://en.wikipedia.org/wiki/Gaussian_function)
"""
function gaussian(p, x)
    A, x0, σ = p[1:3]
    y₀ = length(p) >= 4 ? p[4] : 0.0
    @. A * exp(-((x - x0)^2) / (2σ^2)) + y₀
end

"""
    lorentzian(p, x)

Lorentzian (Cauchy) lineshape with FWHM parameterization.

# Arguments
- `p`: Parameters [A, x₀, Γ] or [A, x₀, Γ, y₀]
  - `A`: Peak amplitude (value at x₀)
  - `x₀`: Center position
  - `Γ`: Full width at half maximum (FWHM)
  - `y₀`: Vertical offset (default: 0.0)
- `x`: Independent variable

The integrated area is `A × π × Γ / 2`.

```math
\\begin{aligned}
    f(x) = \\frac{A}{1 + \\left(\\frac{x - x_0}{\\Gamma/2}\\right)^2} + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Cauchy_distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
"""
function lorentzian(p, x)
    A, x0, Γ = p[1:3]
    y₀ = length(p) >= 4 ? p[4] : 0.0
    @. A / (1 + ((x - x0) / (Γ / 2))^2) + y₀
end

"""
    pseudo_voigt(p, ω)

Pseudo-Voigt profile: a linear combination of Gaussian and Lorentzian functions.

# Arguments
- `p`: Parameters [f₀, ω₀, σ, α]
  - `f₀`: Amplitude
  - `ω₀`: Center position
  - `σ`: Width parameter (HWHM for Lorentzian component)
  - `α`: Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
- `ω`: Independent variable (frequency)

```math
\\begin{aligned}
    f(\\omega) = (1-\\alpha) G(\\omega) + \\alpha L(\\omega)
\\end{aligned}
```

where G and L are normalized Gaussian and Lorentzian functions.

[https://en.wikipedia.org/wiki/Voigt_profile](https://en.wikipedia.org/wiki/Voigt_profile)
"""
function pseudo_voigt(p, ω)
    f_0, ω_0, σ, α = p
    σ_g = σ / sqrt(2 * log(2))
    return @. (1 - α) * f_0 * exp(-(ω - ω_0)^2 / (2 * σ_g^2)) / (σ_g * sqrt(2 * π)) +
        α * f_0 * σ / ((ω - ω_0)^2 + σ^2) / π
end

"""
    gaussian2d(p, coords)

Two-dimensional Gaussian function for curve fitting.

# Arguments
- `p`: Parameters [A, x₀, σ_x, y₀, σ_y] or [A, x₀, σ_x, y₀, σ_y, z₀]
  - `A`: Amplitude
  - `x₀`: Center x-position
  - `σ_x`: Standard deviation in x
  - `y₀`: Center y-position
  - `σ_y`: Standard deviation in y
  - `z₀`: Vertical offset (default: 0.0)
- `coords`: Tuple of (X, Y) coordinate matrices (from meshgrid) or matrix where each row is [x, y]

Returns a vector (flattened) for compatibility with CurveFit.jl.

```math
\\begin{aligned}
    f(x, y) = A \\exp\\left(-\\frac{(x - x_0)^2}{2\\sigma_x^2} - \\frac{(y - y_0)^2}{2\\sigma_y^2}\\right) + z_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function](https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function)
"""
function gaussian2d(p, coords)
    A, x₀, σ_x, y₀, σ_y = p[1:5]
    z₀ = length(p) >= 6 ? p[6] : 0.0

    if coords isa Tuple
        # coords = (X, Y) meshgrid matrices
        X, Y = coords
        result = @. A * exp(-((X - x₀)^2 / (2σ_x^2) + (Y - y₀)^2 / (2σ_y^2))) + z₀
        return vec(result)
    else
        # coords is a matrix where each row is [x, y]
        return @. A * exp(-((coords[:, 1] - x₀)^2 / (2σ_x^2) + (coords[:, 2] - y₀)^2 / (2σ_y^2))) + z₀
    end
end
