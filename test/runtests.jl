using Test
using CurveFit
using Random
using CurveFitModels

Random.seed!(42)

@testset "CurveFitModels Tests" begin

    # =========================================================================
    # PHYSICAL PROPERTY TESTS
    # These tests verify mathematical correctness independent of curve fitting
    # =========================================================================

    @testset "Gaussian physical properties" begin
        A, x0, σ = 3.0, 2.0, 1.0

        # Peak amplitude: maximum value occurs at center and equals A
        @test gaussian([A, x0, σ], [x0])[1] ≈ A

        # Peak amplitude with offset: maximum equals A + y₀
        y₀ = 0.5
        @test gaussian([A, x0, σ, y₀], [x0])[1] ≈ A + y₀

        # Symmetry: f(x₀ - Δx) = f(x₀ + Δx)
        Δx = 0.7
        @test gaussian([A, x0, σ], [x0 - Δx])[1] ≈ gaussian([A, x0, σ], [x0 + Δx])[1]

        # FWHM relationship: at x₀ ± FWHM/2, value should be A/2
        fwhm = sigma_to_fwhm(σ)
        half_max_points = gaussian([A, x0, σ], [x0 - fwhm/2, x0 + fwhm/2])
        @test half_max_points[1] ≈ A / 2 rtol=1e-10
        @test half_max_points[2] ≈ A / 2 rtol=1e-10

        # Asymptotic behavior: decays to offset far from center
        @test gaussian([A, x0, σ, y₀], [x0 + 10σ])[1] ≈ y₀ atol=1e-10

        # 3-parameter form (no offset) defaults to y₀ = 0
        @test gaussian([A, x0, σ], [x0 + 10σ])[1] ≈ 0.0 atol=1e-10
    end

    @testset "Lorentzian physical properties" begin
        A, x0, Γ = 2.0, 1.0, 0.5

        # Peak amplitude: maximum value occurs at center and equals A
        @test lorentzian([A, x0, Γ], [x0])[1] ≈ A

        # Peak amplitude with offset
        y₀ = 0.3
        @test lorentzian([A, x0, Γ, y₀], [x0])[1] ≈ A + y₀

        # Symmetry: f(x₀ - Δx) = f(x₀ + Δx)
        Δx = 0.3
        @test lorentzian([A, x0, Γ], [x0 - Δx])[1] ≈ lorentzian([A, x0, Γ], [x0 + Δx])[1]

        # FWHM definition: at x₀ ± Γ/2, value should be A/2
        half_max_points = lorentzian([A, x0, Γ], [x0 - Γ/2, x0 + Γ/2])
        @test half_max_points[1] ≈ A / 2 rtol=1e-10
        @test half_max_points[2] ≈ A / 2 rtol=1e-10

        # Asymptotic decay: Lorentzian decays as 1/x² (slower than Gaussian)
        # At x = x₀ + 10Γ, value should be A / (1 + 20²) = A/401
        @test lorentzian([A, x0, Γ], [x0 + 10Γ])[1] ≈ A / 401 rtol=1e-10

        # 3-parameter form defaults to y₀ = 0
        @test lorentzian([A, x0, Γ], [x0 + 100Γ])[1] > 0  # Still positive, no offset
    end

    @testset "Exponential decay physical properties" begin
        A, τ, y₀ = 2.0, 1.0, 0.5

        # Initial value: f(0) = A + y₀
        @test single_exponential([A, τ, y₀], [0.0])[1] ≈ A + y₀

        # Asymptotic value: f(∞) → y₀
        @test single_exponential([A, τ, y₀], [100τ])[1] ≈ y₀ atol=1e-10

        # Time constant definition: f(τ) = A/e + y₀
        @test single_exponential([A, τ, y₀], [τ])[1] ≈ A / ℯ + y₀ rtol=1e-10

        # Half-life: f(τ·ln(2)) = A/2 + y₀
        t_half = τ * log(2)
        @test single_exponential([A, τ, y₀], [t_half])[1] ≈ A / 2 + y₀ rtol=1e-10

        # 2-parameter form (no offset) defaults to y₀ = 0
        @test single_exponential([A, τ], [0.0])[1] ≈ A
        @test single_exponential([A, τ], [100τ])[1] ≈ 0.0 atol=1e-10

        # Negative amplitude: exponential rise from offset
        @test single_exponential([-A, τ, y₀], [0.0])[1] ≈ -A + y₀
        @test single_exponential([-A, τ, y₀], [100τ])[1] ≈ y₀ atol=1e-10
    end

    @testset "Sine physical properties" begin
        A, ω, ϕ, y₀ = 1.5, 2.0, 0.0, 0.5

        # Amplitude: max - min = 2A
        t = collect(0.0:0.01:2π/ω)  # One full period
        y = sine([A, ω, ϕ, y₀], t)
        @test maximum(y) - minimum(y) ≈ 2A rtol=0.01

        # Offset: mean value equals y₀
        @test sum(y) / length(y) ≈ y₀ atol=0.01

        # Period: f(t + 2π/ω) = f(t)
        t_test = 0.5
        period = 2π / ω
        @test sine([A, ω, ϕ, y₀], [t_test])[1] ≈ sine([A, ω, ϕ, y₀], [t_test + period])[1]

        # Phase: at ϕ = 0, f(0) = y₀ (starts at zero crossing going up)
        @test sine([A, ω, 0.0, y₀], [0.0])[1] ≈ y₀

        # Phase: at ϕ = π/2, f(0) = A + y₀ (starts at maximum)
        @test sine([A, ω, π/2, y₀], [0.0])[1] ≈ A + y₀

        # 3-parameter form defaults to y₀ = 0
        @test sine([A, ω, 0.0], [0.0])[1] ≈ 0.0
    end

    @testset "Damped sine physical properties" begin
        A, ω, ϕ, τ, y₀ = 2.0, 3.0, 0.0, 1.5, 0.1

        # Initial value: f(0) = A·sin(ϕ) + y₀
        @test damped_sine([A, ω, ϕ, τ, y₀], [0.0])[1] ≈ A * sin(ϕ) + y₀

        # With ϕ = π/2: f(0) = A + y₀
        @test damped_sine([A, ω, π/2, τ, y₀], [0.0])[1] ≈ A + y₀

        # Asymptotic value: f(∞) → y₀
        @test damped_sine([A, ω, ϕ, τ, y₀], [100τ])[1] ≈ y₀ atol=1e-6

        # Envelope decay: peaks follow exponential envelope
        # At t = τ, envelope amplitude is A/e
        # Find a peak near t = τ by checking at t = τ where sin(ωτ + ϕ) ≈ 1
        t_peak = (π/2 - ϕ) / ω  # First peak (where sin = 1)
        if t_peak > 0
            envelope_at_peak = A * exp(-t_peak / τ)
            @test damped_sine([A, ω, ϕ, τ, y₀], [t_peak])[1] ≈ envelope_at_peak + y₀ rtol=0.01
        end

        # 4-parameter form defaults to y₀ = 0
        @test damped_sine([A, ω, ϕ, τ], [100τ])[1] ≈ 0.0 atol=1e-6
    end

    @testset "Stretched exponential physical properties" begin
        A, τ, β, y₀ = 2.0, 1.0, 0.7, 0.5

        # Initial value: f(0) = A + y₀
        @test stretched_exponential([A, τ, β, y₀], [0.0])[1] ≈ A + y₀

        # Asymptotic value: f(∞) → y₀
        @test stretched_exponential([A, τ, β, y₀], [1000.0])[1] ≈ y₀ atol=1e-10

        # β = 1 reduces to single exponential
        t = collect(0.0:0.1:5.0)
        y_stretched = stretched_exponential([A, τ, 1.0, y₀], t)
        y_single = single_exponential([A, τ, y₀], t)
        @test y_stretched ≈ y_single

        # At t = τ with β = 1: f(τ) = A/e + y₀
        @test stretched_exponential([A, τ, 1.0, y₀], [τ])[1] ≈ A / ℯ + y₀ rtol=1e-10

        # Slower decay for β < 1: at t > τ, stretched > single exponential (above offset)
        val_stretched = stretched_exponential([A, τ, 0.5], [2τ])[1]
        val_single = single_exponential([A, τ], [2τ])[1]
        @test val_stretched > val_single

        # 3-parameter form defaults to y₀ = 0
        @test stretched_exponential([A, τ, β], [0.0])[1] ≈ A
        @test stretched_exponential([A, τ, β], [1000.0])[1] ≈ 0.0 atol=1e-10
    end

    @testset "Power law physical properties" begin
        A, n = 2.0, 3.0

        # f(1) = A * 1^n = A
        @test power_law([A, n], [1.0])[1] ≈ A

        # f(0) = 0 for positive n
        @test power_law([A, n], [0.0])[1] ≈ 0.0

        # Known value: A * x^n
        @test power_law([A, n], [2.0])[1] ≈ A * 2.0^n

        # With offset
        y₀ = 0.5
        @test power_law([A, n, y₀], [1.0])[1] ≈ A + y₀
        @test power_law([A, n, y₀], [0.0])[1] ≈ y₀

        # n = 1 is linear
        @test power_law([3.0, 1.0], [5.0])[1] ≈ 15.0

        # n = 0 is constant
        @test power_law([3.0, 0.0], [5.0])[1] ≈ 3.0

        # n = -1 is inverse
        @test power_law([1.0, -1.0], [4.0])[1] ≈ 0.25
    end

    @testset "Logistic physical properties" begin
        L, k, x₀ = 2.0, 3.0, 1.0

        # Midpoint: f(x₀) = L/2
        @test logistic([L, k, x₀], [x₀])[1] ≈ L / 2

        # Asymptotic limits
        @test logistic([L, k, x₀], [100.0])[1] ≈ L atol=1e-10
        @test logistic([L, k, x₀], [-100.0])[1] ≈ 0.0 atol=1e-10

        # With offset
        y₀ = 0.5
        @test logistic([L, k, x₀, y₀], [x₀])[1] ≈ L / 2 + y₀
        @test logistic([L, k, x₀, y₀], [100.0])[1] ≈ L + y₀ atol=1e-10
        @test logistic([L, k, x₀, y₀], [-100.0])[1] ≈ y₀ atol=1e-10

        # Monotonically increasing for positive k
        x = collect(-5.0:0.1:7.0)
        y = logistic([L, k, x₀], x)
        @test all(diff(y) .> 0)

        # Symmetry about midpoint: f(x₀ + Δ) + f(x₀ - Δ) = L
        Δ = 0.5
        @test logistic([L, k, x₀], [x₀ + Δ])[1] + logistic([L, k, x₀], [x₀ - Δ])[1] ≈ L

        # k = 0 gives constant L/2
        @test logistic([L, 0.0, x₀], [0.0])[1] ≈ L / 2

        # Negative k flips the curve (monotonically decreasing)
        y_neg = logistic([L, -k, x₀], x)
        @test all(diff(y_neg) .< 0)
    end

    @testset "Lorentz oscillator physical properties" begin
        A, ν₀, Γ = 1000.0, 100.0, 5.0

        # At resonance (ν = ν₀): real part crosses zero
        # χ = A / (ν₀² - ν² - i·Γ·ν) at ν = ν₀ gives χ = A / (-i·Γ·ν₀)
        # Real part = 0, Imaginary part = -A / (Γ·ν₀)
        χ_at_res = lorentz_oscillator([A, ν₀, Γ], [ν₀])[1]
        @test real(χ_at_res) ≈ 0.0 atol=1e-10
        @test imag(χ_at_res) ≈ A / (Γ * ν₀) rtol=1e-10

        # Real part changes sign at resonance
        ν_below = ν₀ - 2Γ
        ν_above = ν₀ + 2Γ
        χ_below = dielectric_real([A, ν₀, Γ], [ν_below])[1]
        χ_above = dielectric_real([A, ν₀, Γ], [ν_above])[1]
        @test χ_below * χ_above < 0  # Opposite signs

        # Imaginary part is always positive (for positive A, Γ, ν)
        ν_range = collect(50.0:5.0:150.0)
        χ_imag = dielectric_imag([A, ν₀, Γ], ν_range)
        @test all(χ_imag .> 0)

        # Imaginary part peaks near resonance
        idx_max = argmax(χ_imag)
        @test ν_range[idx_max] ≈ ν₀ atol=5.0

        # Low frequency limit: χ(0) = A / ν₀²
        @test lorentz_oscillator([A, ν₀, Γ], [0.0])[1] ≈ A / ν₀^2

        # High frequency limit: χ → 0 as ν → ∞
        χ_high = lorentz_oscillator([A, ν₀, Γ], [10000.0])[1]
        @test abs(χ_high) < 1e-4
    end

    @testset "Pseudo-Voigt limiting cases" begin
        f₀, ω₀, σ = 1.0, 0.0, 1.0
        x = collect(-5.0:0.1:5.0)

        # α = 0: pure Gaussian (area-normalized)
        y_pv_gauss = pseudo_voigt([f₀, ω₀, σ, 0.0], x)
        # Pseudo-Voigt Gaussian component: f₀ * exp(...) / (σ_g * sqrt(2π))
        σ_g = σ / sqrt(2 * log(2))
        y_gauss_normalized = @. f₀ * exp(-(x - ω₀)^2 / (2 * σ_g^2)) / (σ_g * sqrt(2π))
        @test y_pv_gauss ≈ y_gauss_normalized rtol=1e-10

        # α = 1: pure Lorentzian (area-normalized)
        y_pv_lor = pseudo_voigt([f₀, ω₀, σ, 1.0], x)
        # Pseudo-Voigt Lorentzian component: f₀ * σ / ((x - ω₀)² + σ²) / π
        y_lor_normalized = @. f₀ * σ / ((x - ω₀)^2 + σ^2) / π
        @test y_pv_lor ≈ y_lor_normalized rtol=1e-10

        # Intermediate α: should be between pure Gaussian and Lorentzian
        y_pv_mix = pseudo_voigt([f₀, ω₀, σ, 0.5], x)
        # At center, mixed should be between the two extremes
        center_idx = length(x) ÷ 2 + 1
        @test minimum([y_pv_gauss[center_idx], y_pv_lor[center_idx]]) <=
              y_pv_mix[center_idx] <=
              maximum([y_pv_gauss[center_idx], y_pv_lor[center_idx]])
    end

    @testset "Gaussian 2D physical properties" begin
        A, x₀, σ_x, y₀, σ_y, z₀ = 2.0, 0.5, 1.0, -0.3, 1.5, 0.1

        # Test with matrix input (alternative to tuple input)
        coords_matrix = [0.5 -0.3; 0.0 0.0; 1.0 1.0; -1.0 -1.0]
        z = gaussian2d([A, x₀, σ_x, y₀, σ_y, z₀], coords_matrix)
        @test length(z) == 4

        # Peak at center: z(x₀, y₀) = A + z₀
        coords_center = [x₀ y₀]
        z_center = gaussian2d([A, x₀, σ_x, y₀, σ_y, z₀], coords_center)[1]
        @test z_center ≈ A + z₀

        # Separability: f(x,y) = f_x(x) * f_y(y) * A + z₀
        # At (x₀ + σ_x, y₀): f = A * exp(-0.5) * exp(0) + z₀ = A/√e + z₀
        coords_test = [x₀ + σ_x y₀]
        z_test = gaussian2d([A, x₀, σ_x, y₀, σ_y, z₀], coords_test)[1]
        @test z_test ≈ A * exp(-0.5) + z₀ rtol=1e-10

        # Symmetry about center
        coords_sym = [x₀ + 1.0 y₀; x₀ - 1.0 y₀]
        z_sym = gaussian2d([A, x₀, σ_x, y₀, σ_y, z₀], coords_sym)
        @test z_sym[1] ≈ z_sym[2]

        # 5-parameter form defaults to z₀ = 0
        z_no_offset = gaussian2d([A, x₀, σ_x, y₀, σ_y], coords_center)[1]
        @test z_no_offset ≈ A
    end

    # =========================================================================
    # CURVE FITTING TESTS
    # These tests verify that parameters can be recovered from noisy data
    # =========================================================================

    @testset "single_exponential" begin
        # True parameters: A=2.0, τ=0.5, y₀=0.1
        t = collect(0.0:0.05:3.0)
        p_true = [2.0, 0.5, 0.1]
        y_true = single_exponential(p_true, t)
        noise = 0.02 * randn(length(t))
        y_data = y_true .+ noise

        # Fit using CurveFit.jl
        p0 = [1.5, 0.3, 0.0]
        prob = NonlinearCurveFitProblem(single_exponential, p0, t, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.1)
        @test isapprox(p_fit[2], p_true[2], atol=0.1)
        @test isapprox(p_fit[3], p_true[3], atol=0.1)
    end

    @testset "n_exponentials" begin
        t = collect(0.0:0.05:5.0)

        @testset "n=1 matches single_exponential" begin
            mono_exp = n_exponentials(1)
            p = [2.0, 0.5, 0.1]  # [A, τ, y₀]
            y_n = mono_exp(p, t)
            y_single = single_exponential(p, t)
            @test y_n ≈ y_single
        end

        @testset "n=2 mathematical properties" begin
            bi_exp = n_exponentials(2)
            p = [3.0, 0.3, 2.0, 1.5, 0.5]  # [A₁, τ₁, A₂, τ₂, y₀]

            y = bi_exp(p, t)

            # At t=0: y = A₁ + A₂ + y₀
            @test y[1] ≈ p[1] + p[3] + p[5]

            # As t→∞: y → y₀ (check at large t)
            t_large = [100.0]
            y_inf = bi_exp(p, t_large)
            @test isapprox(y_inf[1], p[5], atol=1e-10)

            # Verify it's the sum of two exponentials
            y_manual = @. p[1] * exp(-t / p[2]) + p[3] * exp(-t / p[4]) + p[5]
            @test y ≈ y_manual
        end

        @testset "n=1 fitting" begin
            mono_exp = n_exponentials(1)
            p_true = [2.5, 0.8, 0.2]
            y_data = mono_exp(p_true, t) .+ 0.02 .* randn(length(t))

            p0 = [2.0, 0.5, 0.1]
            prob = NonlinearCurveFitProblem(mono_exp, p0, t, y_data)
            sol = solve(prob)
            p_fit = coef(sol)

            @test isapprox(p_fit[1], p_true[1], atol=0.2)
            @test isapprox(p_fit[2], p_true[2], atol=0.2)
            @test isapprox(p_fit[3], p_true[3], atol=0.1)
        end

        @testset "n=2 fitting (biexponential)" begin
            bi_exp = n_exponentials(2)
            # Fast and slow components with distinct time constants
            p_true = [1.5, 0.2, 1.0, 2.0, 0.1]  # [A₁, τ₁, A₂, τ₂, y₀]
            y_data = bi_exp(p_true, t) .+ 0.02 .* randn(length(t))

            p0 = [1.0, 0.3, 0.8, 1.5, 0.05]
            prob = NonlinearCurveFitProblem(bi_exp, p0, t, y_data)
            sol = solve(prob)
            p_fit = coef(sol)

            # Check total amplitude and offset recovery
            # (individual components may swap due to symmetry)
            total_amp_true = p_true[1] + p_true[3]
            total_amp_fit = p_fit[1] + p_fit[3]
            @test isapprox(total_amp_fit, total_amp_true, atol=0.3)
            @test isapprox(p_fit[5], p_true[5], atol=0.1)

            # Verify fit quality using CurveFit's rss (residual sum of squares)
            @test sqrt(rss(sol) / length(t)) < 0.05
        end

        @testset "n=3 triexponential" begin
            tri_exp = n_exponentials(3)
            p = [1.0, 0.1, 2.0, 0.5, 1.5, 2.0, 0.3]  # [A₁, τ₁, A₂, τ₂, A₃, τ₃, y₀]

            y = tri_exp(p, t)

            # At t=0: y = A₁ + A₂ + A₃ + y₀
            @test y[1] ≈ p[1] + p[3] + p[5] + p[7]

            # Verify manual calculation
            y_manual = @. p[1] * exp(-t / p[2]) + p[3] * exp(-t / p[4]) + p[5] * exp(-t / p[6]) + p[7]
            @test y ≈ y_manual
        end
    end

    @testset "stretched_exponential" begin
        # Start at t > 0 to avoid derivative singularity at t=0 for β < 1
        rng = Random.MersenneTwister(99)
        t = collect(0.01:0.05:10.0)
        p_true = [2.0, 1.0, 0.8, 0.1]  # [A, τ, β, y₀]
        y_true = stretched_exponential(p_true, t)
        noise = 0.01 * randn(rng, length(t))
        y_data = y_true .+ noise

        p0 = [2.2, 1.2, 0.9, 0.05]
        prob = NonlinearCurveFitProblem(stretched_exponential, p0, t, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.5)
        @test isapprox(p_fit[4], p_true[4], atol=0.2)
        @test sqrt(rss(sol) / length(t)) < 0.05
    end

    @testset "gaussian" begin
        # True parameters: A=3.0, x0=2.0, σ=0.5, y₀=0.2
        x = collect(-1.0:0.1:5.0)
        p_true = [3.0, 2.0, 0.5, 0.2]
        y_true = gaussian(p_true, x)
        noise = 0.05 * randn(length(x))
        y_data = y_true .+ noise

        p0 = [2.5, 1.8, 0.4, 0.0]
        prob = NonlinearCurveFitProblem(gaussian, p0, x, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.2)
        @test isapprox(p_fit[2], p_true[2], atol=0.2)
        @test isapprox(p_fit[3], p_true[3], atol=0.2)
        @test isapprox(p_fit[4], p_true[4], atol=0.2)
    end

    @testset "sigma_fwhm_conversion" begin
        # Test round-trip conversion
        σ = 1.0
        fwhm = sigma_to_fwhm(σ)
        @test isapprox(fwhm, 2.3548, atol=0.001)
        @test isapprox(fwhm_to_sigma(fwhm), σ, atol=1e-10)

        # Test known relationship: FWHM = 2√(2ln2) × σ
        @test isapprox(fwhm, 2 * sqrt(2 * log(2)) * σ, atol=1e-10)
    end

    @testset "area_functions" begin
        # Gaussian area: numerical integration vs analytical formula
        A, σ = 2.0, 1.5
        x = collect(range(-20, 20, length=10000))
        dx = x[2] - x[1]
        y = gaussian([A, 0.0, σ], x)
        numerical_area = sum(y) * dx
        @test isapprox(numerical_area, gaussian_area(A, σ), rtol=0.001)

        # Lorentzian area: numerical integration vs analytical formula
        A, Γ = 3.0, 2.0
        x = collect(range(-500, 500, length=100000))
        dx = x[2] - x[1]
        y = lorentzian([A, 0.0, Γ], x)
        numerical_area = sum(y) * dx
        @test isapprox(numerical_area, lorentzian_area(A, Γ), rtol=0.01)
    end

    @testset "poly" begin
        # Constant
        @test poly([5.0], [1.0, 2.0, 3.0]) ≈ [5.0, 5.0, 5.0]

        # Linear: 1 + 2x
        @test poly([1.0, 2.0], [0.0, 1.0, 2.0]) ≈ [1.0, 3.0, 5.0]

        # Quadratic: 1 + 2x + 3x²
        @test poly([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) ≈ [1.0, 6.0, 17.0]
    end

    @testset "combine" begin
        model = combine(lorentzian, 3, poly, 2)

        x = collect(-5.0:0.1:5.0)
        p = [3.0, 0.0, 1.0, 0.5, 0.1]  # [A, x0, Γ, c0, c1]

        y_lor = lorentzian(p[1:3], x)
        y_poly = poly(p[4:5], x)
        y_combined = model(p, x)

        @test y_combined ≈ y_lor .+ y_poly

        # Test fitting with combined model
        y_data = y_combined .+ 0.01 .* randn(length(x))
        p0 = [2.5, 0.1, 1.2, 0.4, 0.05]
        prob = NonlinearCurveFitProblem(model, p0, x, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p[1], atol=0.3)
        @test isapprox(p_fit[2], p[2], atol=0.2)
        @test isapprox(p_fit[3], p[3], atol=0.2)
    end

    @testset "lorentzian" begin
        # True parameters: A=2.0, x0=0.5, Γ=0.5, y₀=0.1
        x = collect(-2.0:0.05:3.0)
        p_true = [2.0, 0.5, 0.5, 0.1]
        y_true = lorentzian(p_true, x)
        noise = 0.02 * randn(length(x))
        y_data = y_true .+ noise

        p0 = [1.5, 0.3, 0.4, 0.0]
        prob = NonlinearCurveFitProblem(lorentzian, p0, x, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.3)
        @test isapprox(p_fit[2], p_true[2], atol=0.2)
        @test isapprox(p_fit[3], p_true[3], atol=0.2)
        @test isapprox(p_fit[4], p_true[4], atol=0.2)
    end

    @testset "pseudo_voigt" begin
        # True parameters: f_0=2.0, ω_0=1.0, σ=0.3, α=0.5
        ω = collect(-2.0:0.05:4.0)
        p_true = [2.0, 1.0, 0.3, 0.5]
        y_true = pseudo_voigt(p_true, ω)
        noise = 0.02 * randn(length(ω))
        y_data = y_true .+ noise

        p0 = [1.5, 0.8, 0.25, 0.4]
        prob = NonlinearCurveFitProblem(pseudo_voigt, p0, ω, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.3)
        @test isapprox(p_fit[2], p_true[2], atol=0.2)
        @test isapprox(p_fit[3], p_true[3], atol=0.2)
        @test isapprox(p_fit[4], p_true[4], atol=0.2)
    end

    @testset "power_law" begin
        x = collect(0.5:0.1:5.0)
        p_true = [2.0, 1.5, 0.3]  # [A, n, y₀]
        y_true = power_law(p_true, x)
        noise = 0.05 * randn(length(x))
        y_data = y_true .+ noise

        p0 = [1.5, 1.2, 0.0]
        prob = NonlinearCurveFitProblem(power_law, p0, x, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.3)
        @test isapprox(p_fit[2], p_true[2], atol=0.3)
        @test isapprox(p_fit[3], p_true[3], atol=0.3)
    end

    @testset "logistic" begin
        x = collect(-5.0:0.1:7.0)
        p_true = [2.0, 1.5, 1.0, 0.1]  # [L, k, x₀, y₀]
        y_true = logistic(p_true, x)
        noise = 0.02 * randn(length(x))
        y_data = y_true .+ noise

        p0 = [1.8, 1.2, 0.8, 0.0]
        prob = NonlinearCurveFitProblem(logistic, p0, x, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.3)
        @test isapprox(p_fit[2], p_true[2], atol=0.3)
        @test isapprox(p_fit[3], p_true[3], atol=0.3)
        @test isapprox(p_fit[4], p_true[4], atol=0.2)
    end

    @testset "sine" begin
        # True parameters: A=1.5, ω=2.0, ϕ=0.5, y₀=0.2
        t = collect(0.0:0.05:4π)
        p_true = [1.5, 2.0, 0.5, 0.2]
        y_true = sine(p_true, t)
        noise = 0.05 * randn(length(t))
        y_data = y_true .+ noise

        p0 = [1.3, 1.9, 0.3, 0.0]
        prob = NonlinearCurveFitProblem(sine, p0, t, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.2)
        @test isapprox(p_fit[2], p_true[2], atol=0.2)
    end

    @testset "damped_sine" begin
        # True parameters: A=2.0, ω=3.0, ϕ=0.2, τ=1.5, y₀=0.1
        t = collect(0.0:0.05:5.0)
        p_true = [2.0, 3.0, 0.2, 1.5, 0.1]
        y_true = damped_sine(p_true, t)
        noise = 0.03 * randn(length(t))
        y_data = y_true .+ noise

        p0 = [1.8, 2.8, 0.1, 1.2, 0.0]
        prob = NonlinearCurveFitProblem(damped_sine, p0, t, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.3)
        @test isapprox(p_fit[2], p_true[2], atol=0.3)
    end

    @testset "poly" begin
        # Fit a quadratic polynomial: c₀ + c₁x + c₂x²
        x = collect(-3.0:0.2:3.0)
        p_true = [1.5, -0.8, 0.3]  # [c₀, c₁, c₂]
        y_true = poly(p_true, x)
        noise = 0.02 * randn(length(x))
        y_data = y_true .+ noise

        p0 = [1.0, -0.5, 0.2]
        prob = NonlinearCurveFitProblem(poly, p0, x, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.2)
        @test isapprox(p_fit[2], p_true[2], atol=0.2)
        @test isapprox(p_fit[3], p_true[3], atol=0.2)
    end

    @testset "lorentz_oscillator" begin
        # Verify that dielectric_real and dielectric_imag are the real/imag parts
        p = [1.0, 100.0, 10.0]  # A, ν₀, Γ
        ν = collect(80.0:5.0:120.0)

        χ_complex = lorentz_oscillator(p, ν)
        χ_real = dielectric_real(p, ν)
        χ_imag = dielectric_imag(p, ν)

        @test all(isapprox.(real.(χ_complex), χ_real, atol=1e-12))
        @test all(isapprox.(imag.(χ_complex), χ_imag, atol=1e-12))
    end

    @testset "gaussian2d" begin
        # Create meshgrid
        x = collect(-3.0:0.5:3.0)
        y = collect(-3.0:0.5:3.0)
        X = [xi for xi in x, _ in y]
        Y = [yj for _ in x, yj in y]
        coords = (X, Y)

        # True parameters: A=2.0, x₀=0.5, σ_x=1.0, y₀=-0.5, σ_y=1.2, z₀=0.1
        p_true = [2.0, 0.5, 1.0, -0.5, 1.2, 0.1]
        z_true = gaussian2d(p_true, coords)
        noise = 0.02 * randn(length(z_true))
        z_data = z_true .+ noise

        # Create wrapper function with correct signature for CurveFit.jl
        model(p, _) = gaussian2d(p, coords)

        p0 = [1.8, 0.3, 0.8, -0.3, 1.0, 0.0]
        prob = NonlinearCurveFitProblem(model, p0, 1:length(z_data), z_data)
        sol = solve(prob)
        p_fit = coef(sol)

        @test isapprox(p_fit[1], p_true[1], atol=0.3)
        @test isapprox(p_fit[2], p_true[2], atol=0.3)
        @test isapprox(p_fit[3], p_true[3], atol=0.3)
        @test isapprox(p_fit[4], p_true[4], atol=0.3)
        @test isapprox(p_fit[5], p_true[5], atol=0.3)
    end

    @testset "dielectric_imag fitting" begin
        # Fit absorption spectrum using dielectric_imag (Lorentz oscillator)
        ν = collect(80.0:0.5:120.0)
        p_true = [500.0, 100.0, 3.0]  # [A, ν₀, Γ]
        y_true = dielectric_imag(p_true, ν)
        noise = 0.0001 * randn(length(ν))
        y_data = y_true .+ noise

        p0 = [400.0, 98.0, 4.0]
        prob = NonlinearCurveFitProblem(dielectric_imag, p0, ν, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        # Center frequency should be well-recovered
        @test isapprox(p_fit[2], p_true[2], atol=1.0)
        # Linewidth should be approximately correct
        @test isapprox(p_fit[3], p_true[3], atol=1.0)
    end

    @testset "dielectric_real fitting" begin
        # Fit dispersive lineshape using dielectric_real
        # Use finer grid and lower noise since dispersive lineshape is more sensitive
        ν = collect(80.0:0.2:120.0)
        p_true = [500.0, 100.0, 3.0]  # [A, ν₀, Γ]
        y_true = dielectric_real(p_true, ν)
        rng = MersenneTwister(42)
        noise = 0.001 * randn(rng, length(ν))
        y_data = y_true .+ noise

        p0 = [450.0, 99.0, 3.5]
        prob = NonlinearCurveFitProblem(dielectric_real, p0, ν, y_data)
        sol = solve(prob)
        p_fit = coef(sol)

        # Center frequency should be well-recovered
        @test isapprox(p_fit[2], p_true[2], atol=1.0)
        # Linewidth should be approximately correct
        @test isapprox(p_fit[3], p_true[3], atol=1.0)
    end

    # =========================================================================
    # EDGE CASES AND NUMERICAL TESTS
    # =========================================================================

    @testset "Single point evaluation" begin
        # All functions should work with single-element arrays
        @test length(gaussian([1.0, 0.0, 1.0], [0.0])) == 1
        @test length(lorentzian([1.0, 0.0, 1.0], [0.0])) == 1
        @test length(single_exponential([1.0, 1.0], [0.0])) == 1
        @test length(stretched_exponential([1.0, 1.0, 0.5], [0.0])) == 1
        @test length(sine([1.0, 1.0, 0.0], [0.0])) == 1
        @test length(damped_sine([1.0, 1.0, 0.0, 1.0], [0.0])) == 1
        @test length(pseudo_voigt([1.0, 0.0, 1.0, 0.5], [0.0])) == 1
        @test length(power_law([1.0, 2.0], [1.0])) == 1
        @test length(logistic([1.0, 1.0, 0.0], [0.0])) == 1
        @test length(lorentz_oscillator([1.0, 100.0, 1.0], [100.0])) == 1
        @test length(dielectric_real([1.0, 100.0, 1.0], [100.0])) == 1
        @test length(dielectric_imag([1.0, 100.0, 1.0], [100.0])) == 1
        @test length(poly([1.0, 2.0], [0.0])) == 1
    end

    @testset "Empty array handling" begin
        # Functions should return empty arrays for empty input
        @test length(gaussian([1.0, 0.0, 1.0], Float64[])) == 0
        @test length(lorentzian([1.0, 0.0, 1.0], Float64[])) == 0
        @test length(single_exponential([1.0, 1.0], Float64[])) == 0
        @test length(stretched_exponential([1.0, 1.0, 0.5], Float64[])) == 0
        @test length(power_law([1.0, 2.0], Float64[])) == 0
        @test length(logistic([1.0, 1.0, 0.0], Float64[])) == 0
    end

    @testset "poly higher degrees" begin
        # Cubic: 1 - x + 2x² + 3x³
        x = [0.0, 1.0, 2.0, -1.0]
        p = [1.0, -1.0, 2.0, 3.0]
        expected = @. 1.0 - x + 2x^2 + 3x^3
        @test poly(p, x) ≈ expected

        # Quartic with negative coefficients
        p_quartic = [1.0, 0.0, -1.0, 0.0, 1.0]  # 1 - x² + x⁴
        expected_quartic = @. 1.0 - x^2 + x^4
        @test poly(p_quartic, x) ≈ expected_quartic
    end

    @testset "Area functions with multiple parameter sets" begin
        # Test area formulas with various parameter combinations
        for A in [0.5, 1.0, 2.0, 5.0]
            for width in [0.1, 1.0, 3.0]
                # Gaussian: Area = A × σ × √(2π)
                @test gaussian_area(A, width) ≈ A * width * sqrt(2π)

                # Lorentzian: Area = A × π × Γ / 2
                @test lorentzian_area(A, width) ≈ A * π * width / 2
            end
        end
    end

    @testset "n_exponentials edge cases" begin
        t = collect(0.0:0.1:5.0)

        # n=0 should just return offset (edge case)
        # Note: This tests the implementation handles n=0 gracefully
        zero_exp = n_exponentials(0)
        p = [0.5]  # Just offset
        y = zero_exp(p, t)
        @test all(y .≈ 0.5)

        # Large n (n=5) should still work
        penta_exp = n_exponentials(5)
        p5 = [1.0, 0.1, 0.8, 0.3, 0.6, 0.5, 0.4, 1.0, 0.2, 2.0, 0.1]
        y5 = penta_exp(p5, t)
        # At t=0: sum of all amplitudes + offset
        @test y5[1] ≈ sum(p5[1:2:9]) + p5[11]
    end

    @testset "Numerical stability" begin
        # Very large x values for Gaussian (should not overflow)
        y = gaussian([1.0, 0.0, 1.0], [1000.0])
        @test y[1] ≈ 0.0 atol=1e-100
        @test isfinite(y[1])

        # Very small time constant for exponential
        y = single_exponential([1.0, 1e-10, 0.0], [1.0])
        @test y[1] ≈ 0.0 atol=1e-100
        @test isfinite(y[1])

        # Lorentzian far from center
        y = lorentzian([1.0, 0.0, 1.0], [1e6])
        @test isfinite(y[1])
        @test y[1] > 0
    end

    @testset "Float32 support" begin
        x32 = Float32.(0.0:0.1:4.0)

        # Gaussian
        p = Float32[3.0, 2.0, 0.5]
        y = gaussian(p, x32)
        @test eltype(y) == Float32

        # Gaussian with offset
        y = gaussian(Float32[3.0, 2.0, 0.5, 0.1], x32)
        @test eltype(y) == Float32

        # Lorentzian
        y = lorentzian(Float32[2.0, 1.0, 0.5], x32)
        @test eltype(y) == Float32

        # single_exponential
        y = single_exponential(Float32[2.0, 0.5], x32)
        @test eltype(y) == Float32

        # stretched_exponential
        y = stretched_exponential(Float32[2.0, 0.5, 0.7], x32)
        @test eltype(y) == Float32

        # sine
        y = sine(Float32[1.0, 2.0, 0.0], x32)
        @test eltype(y) == Float32

        # damped_sine
        y = damped_sine(Float32[1.0, 2.0, 0.0, 1.0], x32)
        @test eltype(y) == Float32

        # n_exponentials
        mono = n_exponentials(1)
        y = mono(Float32[2.0, 0.5, 0.1], x32)
        @test eltype(y) == Float32

        # poly
        y = poly(Float32[1.0, 2.0], x32)
        @test eltype(y) == Float32

        # power_law
        x32_pos = Float32.(0.1:0.1:4.0)
        y = power_law(Float32[2.0, 1.5], x32_pos)
        @test eltype(y) == Float32

        # logistic
        y = logistic(Float32[1.0, 1.0, 0.0], x32)
        @test eltype(y) == Float32
    end

    @testset "n_exponentials input validation" begin
        @test_throws ArgumentError n_exponentials(-1)
        @test_throws ArgumentError n_exponentials(-5)
    end

end
