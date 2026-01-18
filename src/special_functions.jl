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
    A, ν₀, Γ = p
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
    A, ν₀, Γ = p
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
    A, ν₀, Γ = p
    return @. A * Γ * ν / ((ν^2 - ν₀^2)^2 + Γ^2 * ν^2)
end

"""
    cavity_mode_energy(p, θ)

Cavity mode energy as a function of incident angle.

# Arguments
- `p`: Parameters [E₀, n_eff]
  - `E₀`: Cavity mode energy at normal incidence
  - `n_eff`: Effective refractive index
- `θ`: Incident angle(s) in radians

```math
\\begin{aligned}
    E_\\text{cavity}(\\theta) = \\frac{E_0}{\\sqrt{1 - \\sin^2(\\theta)/n_{\\text{eff}}^2}}
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Fabry–Pérot_interferometer](https://en.wikipedia.org/wiki/Fabry–Pérot_interferometer)
"""
function cavity_mode_energy(p, θ)
    E₀, n_eff = p
    return @. E₀ / sqrt(1 - (sin(θ) / n_eff)^2)
end

"""
    polariton_branches(θ, E₀, E_v, n_eff, Ω; branch=:upper)

Polariton dispersion from the coupled-oscillator model.

Computes the upper (`:upper`) or lower (`:lower`) polariton branch energy
as a function of angle by diagonalizing the coupled Hamiltonian.

# Arguments
- `θ`: Incident angle(s) in radians
- `E₀`: Cavity mode energy at normal incidence
- `E_v`: Vibrational/excitonic energy
- `n_eff`: Effective refractive index
- `Ω`: Rabi splitting energy

# Keyword Arguments
- `branch`: `:upper` or `:lower` (default: `:upper`)

```math
\\begin{aligned}
    E_{\\pm}(\\theta) = \\frac{1}{2}\\left(E_c(\\theta) + E_v\\right) \\pm \\frac{1}{2} \\sqrt{\\Omega^2 + (E_c(\\theta) - E_v)^2}
\\end{aligned}
```

For fitting both branches simultaneously, use a wrapper:
```julia
function both_branches(p, θ)
    E₀, E_v, n_eff, Ω = p
    upper = polariton_branches(θ, E₀, E_v, n_eff, Ω; branch=:upper)
    lower = polariton_branches(θ, E₀, E_v, n_eff, Ω; branch=:lower)
    vcat(upper, lower)
end
```

[https://en.wikipedia.org/wiki/Polariton](https://en.wikipedia.org/wiki/Polariton)
"""
function polariton_branches(θ, E₀, E_v, n_eff, Ω; branch=:upper)
    E_c = cavity_mode_energy([E₀, n_eff], θ)
    sign = branch == :upper ? 1 : -1
    return @. 0.5 * (E_v + E_c + sign * sqrt(Ω^2 + (E_c - E_v)^2))
end


"""
    cavity_transmittance(p, ν)

Fabry-Pérot cavity transmittance as a function of frequency.

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
    n, α, L, R, ϕ = p
    T = 1 - R
    @. T^2 * exp(-α * L) / (1 + R^2 * exp(-2 * α * L) - 2 * R * exp(-α * L) * cos(4π * n * L * ν + 2 * ϕ))
end
