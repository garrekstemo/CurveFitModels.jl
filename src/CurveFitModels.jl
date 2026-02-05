module CurveFitModels

export single_exponential,
       stretched_exponential,
       n_exponentials,
       gaussian,
       gaussian_area,
       gaussian2d,
       lorentzian,
       lorentzian_area,
       sigma_to_fwhm,
       fwhm_to_sigma,
       poly,
       combine,
       sine,
       damped_sine,
       pseudo_voigt,
       power_law,
       logistic,
       lorentz_oscillator,
       dielectric_real,
       dielectric_imag

include("lineshapes.jl")
include("temporal.jl")

end # module
