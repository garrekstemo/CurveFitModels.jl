"""
    complex_lorentzian(p, ν)

Complex Lorentzian susceptibility function.

# Arguments
- `p`: Parameters [A, ν₀, Γ]
  - `A`: Oscillator strength
  - `ν₀`: Resonance frequency
  - `Γ`: Linewidth (damping)
- `ν`: Frequency (independent variable)

```math
\\begin{aligned}
\\chi(\\nu) = \\frac{A}{\\nu_0^2 - \\nu^2 - i\\Gamma\\nu}
\\end{aligned}
```

The real and imaginary parts are `dielectric_real` and `dielectric_imag`, respectively.

[https://en.wikipedia.org/wiki/Lorentz_oscillator_model](https://en.wikipedia.org/wiki/Lorentz_oscillator_model)
"""
function complex_lorentzian(p, ν)
    A, ν₀, Γ = p[1], p[2], p[3]
    return @. A / (ν₀^2 - ν^2 - im * Γ * ν)
end

"""
    dielectric_real(p, ν)

Real part of complex Lorentzian susceptibility (Lorentz oscillator model).

This is equivalent to `real(complex_lorentzian(p, ν))`.

# Arguments
- `p`: Parameters [A, ν₀, Γ]
  - `A`: Oscillator strength
  - `ν₀`: Resonance frequency
  - `Γ`: Linewidth (damping)
- `ν`: Frequency (independent variable)

```math
\\begin{aligned}
\\chi_1(\\nu) = \\text{Re}\\left[\\frac{A}{\\nu_0^2 - \\nu^2 - i\\Gamma\\nu}\\right] = \\frac{A (\\nu_0^2 - \\nu^2)}{(\\nu_0^2 - \\nu^2)^2 + (\\Gamma\\nu)^2}
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Lorentz_oscillator_model](https://en.wikipedia.org/wiki/Lorentz_oscillator_model)
"""
function dielectric_real(p, ν)
    A, ν₀, Γ = p[1], p[2], p[3]
    return @. A * (ν₀^2 - ν^2) / ((ν^2 - ν₀^2)^2 + Γ^2 * ν^2)
end

"""
    dielectric_imag(p, ν)

Imaginary part of complex Lorentzian susceptibility (Lorentz oscillator model).

This is equivalent to `imag(complex_lorentzian(p, ν))`.

# Arguments
- `p`: Parameters [A, ν₀, Γ]
  - `A`: Oscillator strength
  - `ν₀`: Resonance frequency
  - `Γ`: Linewidth (damping)
- `ν`: Frequency (independent variable)

```math
\\begin{aligned}
\\chi_2(\\nu) = \\text{Im}\\left[\\frac{A}{\\nu_0^2 - \\nu^2 - i\\Gamma\\nu}\\right] = \\frac{A \\Gamma \\nu}{(\\nu_0^2 - \\nu^2)^2 + (\\Gamma\\nu)^2}
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Lorentz_oscillator_model](https://en.wikipedia.org/wiki/Lorentz_oscillator_model)
"""
function dielectric_imag(p, ν)
    A, ν₀, Γ = p[1], p[2], p[3]
    return @. A * Γ * ν / ((ν^2 - ν₀^2)^2 + Γ^2 * ν^2)
end

"""
    cavity_transmittance(p, ν)

Fabry-Pérot cavity transmittance with an absorbing medium as a function of frequency.

# Arguments
- `p`: Parameters [n, α, L, R, ϕ]
  - `n`: Refractive index
  - `α`: Absorption coefficient
  - `L`: Cavity length
  - `R`: Mirror reflectance (T = 1 - R assumed)
  - `ϕ`: Phase shift upon reflection
- `ν`: Frequency (independent variable)

```math
\\begin{aligned}
    T(\\nu) = \\frac{(1-R)^2 e^{-\\alpha L}}{1 + R^2 e^{-2\\alpha L} - 2R e^{-\\alpha L} \\cos(4\\pi n L \\nu + 2\\phi)}
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Fabry%E2%80%93P%C3%A9rot_interferometer](https://en.wikipedia.org/wiki/Fabry%E2%80%93P%C3%A9rot_interferometer)
"""
function cavity_transmittance(p, ν)
    n, α, L, R, ϕ = p[1], p[2], p[3], p[4], p[5]
    T = 1 - R
    # Precompute exponential to avoid redundant calculations
    e = exp(-α * L)
    @. T^2 * e / (1 + R^2 * e^2 - 2 * R * e * cos(4π * n * L * ν + 2 * ϕ))
end
