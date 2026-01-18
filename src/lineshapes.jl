"""
    gaussian(p, x)

Gaussian function with standard deviation parameterization.

# Arguments
- `p`: Parameters [A, x₀, σ] or [A, x₀, σ, y₀]
  - `A`: Amplitude
  - `x₀`: Center position
  - `σ`: Standard deviation (σ = FWHM / 2.355)
  - `y₀`: Vertical offset (default: 0.0)
- `x`: Independent variable

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
    gaussian_fwhm(p, x)

Gaussian function with FWHM (Full Width at Half Maximum) parameterization.

# Arguments
- `p`: Parameters [A, x₀, Γ] or [A, x₀, Γ, y₀]
  - `A`: Amplitude
  - `x₀`: Center position
  - `Γ`: Full width at half maximum (FWHM)
  - `y₀`: Vertical offset (default: 0.0)
- `x`: Independent variable

```math
\\begin{aligned}
    f(x) = A \\exp\\left(-4 \\ln(2) \\frac{(x - x_0)^2}{\\Gamma^2}\\right) + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Gaussian_function](https://en.wikipedia.org/wiki/Gaussian_function)
"""
function gaussian_fwhm(p, x)
    A, x0, Γ = p[1:3]
    y₀ = length(p) >= 4 ? p[4] : 0.0
    @. A * exp(-4 * log(2) * ((x - x0)^2) / (Γ^2)) + y₀
end

"""
    lorentzian(p, x)

Lorentzian (Cauchy) distribution function.

# Arguments
- `p`: Parameters [A, x₀, γ] or [A, x₀, γ, y₀]
  - `A`: Amplitude scaling factor
  - `x₀`: Center position
  - `γ`: Half-width at half maximum (HWHM)
  - `y₀`: Vertical offset (default: 0.0)
- `x`: Independent variable

```math
\\begin{aligned}
    f(x) = \\frac{A \\gamma}{(x - x_0)^2 + \\gamma^2} + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Cauchy_distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
"""
function lorentzian(p, x)
    A, x0, σ = p[1:3]
    y₀ = length(p) >= 4 ? p[4] : 0.0
    @. A * σ / ((x - x0)^2 + σ^2) + y₀
end

"""
    lorentzian_fwhm(p, x)

Lorentzian (Cauchy) function with FWHM parameterization.

# Arguments
- `p`: Parameters [A, x₀, Γ] or [A, x₀, Γ, y₀]
  - `A`: Amplitude (area under curve × π)
  - `x₀`: Center position
  - `Γ`: Full width at half maximum (FWHM)
  - `y₀`: Vertical offset (default: 0.0)
- `x`: Independent variable

```math
\\begin{aligned}
    f(x) = \\frac{A}{\\pi} \\frac{\\Gamma/2}{(x - x_0)^2 + (\\Gamma/2)^2} + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Cauchy_distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
"""
function lorentzian_fwhm(p, x)
    A, x0, Γ = p[1:3]
    y₀ = length(p) >= 4 ? p[4] : 0.0
    @. (A / π) * (Γ / 2) / ((x - x0)^2 + (Γ / 2)^2) + y₀
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
