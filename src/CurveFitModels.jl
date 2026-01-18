module CurveFitModels

export single_exponential,
       gaussian,
       gaussian_fwhm,
       gaussian2d,
       lorentzian,
       lorentzian_fwhm,
       sine,
       damped_sine,
       pseudo_voigt,
       complex_lorentzian,
       dielectric_real,
       dielectric_imag,
       cavity_mode_energy,
       polariton_branches,
       cavity_transmittance,
       fsr

include("lineshapes.jl")
include("temporal.jl")
include("special_functions.jl")

end # module
