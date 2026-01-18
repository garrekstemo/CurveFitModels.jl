"""
    single_exponential(p, t)

Exponential decay function.

# Arguments
- `p`: Parameters [A, τ] or [A, τ, y₀]
  - `A`: Amplitude
  - `τ`: Decay time constant
  - `y₀`: Vertical offset (default: 0.0)
- `t`: Independent variable (time)

```math
\\begin{aligned}
    f(t) = A e^{-t/\\tau} + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Exponential_decay](https://en.wikipedia.org/wiki/Exponential_decay)
"""
function single_exponential(p, t)
    A, τ = p[1:2]
    y₀ = length(p) >= 3 ? p[3] : 0.0
    @. A * exp(-t / τ) + y₀
end

"""
    sine(p, t)

Sinusoidal function.

# Arguments
- `p`: Parameters [A, ω, ϕ] or [A, ω, ϕ, y₀]
  - `A`: Amplitude
  - `ω`: Angular frequency
  - `ϕ`: Phase
  - `y₀`: Vertical offset (default: 0.0)
- `t`: Independent variable (time)

```math
\\begin{aligned}
    f(t) = A \\sin(\\omega t + \\phi) + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Sine_wave](https://en.wikipedia.org/wiki/Sine_wave)
"""
function sine(p, t)
    A, ω, ϕ = p[1:3]
    y₀ = length(p) >= 4 ? p[4] : 0.0
    return @. A * sin(t * ω + ϕ) + y₀
end

"""
    damped_sine(p, t)

Damped sine function (exponentially decaying sinusoid).

# Arguments
- `p`: Parameters [A, ω, ϕ, τ] or [A, ω, ϕ, τ, y₀]
  - `A`: Amplitude
  - `ω`: Angular frequency
  - `ϕ`: Phase
  - `τ`: Decay time constant
  - `y₀`: Vertical offset (default: 0.0)
- `t`: Independent variable (time)

```math
\\begin{aligned}
    f(t) = A e^{-t/\\tau} \\sin(\\omega t + \\phi) + y_0
\\end{aligned}
```

[https://en.wikipedia.org/wiki/Damping](https://en.wikipedia.org/wiki/Damping)
"""
function damped_sine(p, t)
    A, ω, ϕ, τ = p[1:4]
    y₀ = length(p) >= 5 ? p[5] : 0.0
    return @. A * exp(-t / τ) * sin(t * ω + ϕ) + y₀
end
