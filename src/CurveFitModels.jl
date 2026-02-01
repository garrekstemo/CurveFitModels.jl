module CurveFitModels

export single_exponential,
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
       complex_lorentzian,
       dielectric_real,
       dielectric_imag,
       cavity_transmittance

include("lineshapes.jl")
include("temporal.jl")
include("special_functions.jl")

end # module
