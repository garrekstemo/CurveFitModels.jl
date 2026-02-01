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

"""
    n_exponentials(n::Int)

Generate a model function for a sum of `n` exponential decays.

Returns a function `model(p, t)` where:
- `p`: Parameters [A₁, τ₁, A₂, τ₂, ..., Aₙ, τₙ, y₀]
  - `Aᵢ`: Amplitude of i-th exponential
  - `τᵢ`: Time constant of i-th exponential
  - `y₀`: Offset (final parameter)
- `t`: Independent variable (time)

```math
f(t) = \\sum_{i=1}^{n} A_i e^{-t/\\tau_i} + y_0
```

# Example
```julia
bi_exp = n_exponentials(2)
p0 = [A1, τ1, A2, τ2, offset]
sol = solve(NonlinearCurveFitProblem(bi_exp, p0, t, y))
```
"""
function n_exponentials(n::Int)
    function model(p, t)
        y₀ = p[end]
        y = similar(p, length(t))
        fill!(y, y₀)
        for i in 1:n
            A = p[2i - 1]
            τ = p[2i]
            @. y += A * exp(-t / τ)
        end
        return y
    end
    return model
end
