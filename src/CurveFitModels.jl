module CurveFitModels

export single_exponential,
       gaussian,
       gaussian2d,
       lorentzian,
       sigma_to_fwhm,
       fwhm_to_sigma,
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
