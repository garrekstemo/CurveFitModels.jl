using Documenter, CurveFitModels

makedocs(;
    sitename = "CurveFitModels.jl",
    authors = "Garrek Stemo",
    modules = [CurveFitModels],
    pages = [
        "Home" => "index.md",
        "Lineshapes" => "lineshapes.md",
        "Temporal" => "temporal.md",
        "Utilities" => "utilities.md",
    ],
)

deploydocs(;
    repo = "github.com/garrekstemo/CurveFitModels.jl.git",
)
