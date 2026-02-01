using Test
using CurveFit
using Random

include("../src/CurveFitModels.jl")
using .CurveFitModels

Random.seed!(42)

@testset "CurveFitModels Tests" begin

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

    @testset "complex_lorentzian" begin
        # Verify that dielectric_real and dielectric_imag are the real/imag parts
        p = [1.0, 100.0, 10.0]  # A, ν₀, Γ
        ν = collect(80.0:5.0:120.0)

        χ_complex = complex_lorentzian(p, ν)
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

end
